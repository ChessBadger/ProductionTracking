import re
import os
import glob
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # For improved date handling
from fpdf import FPDF
from adjustText import adjust_text

# ---------------------------
# STEP 1: Data Extraction
# ---------------------------
# Set the directory where your production PDF files are stored
pdf_dir = r"C:\Users\Laptop 122\Desktop\Store Prep\06 Employee Reports\01 Old\Financial"  # update this to your folder with PDFs
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

all_rows = []

for file in pdf_files:
    # Extract the 6-digit date from the filename (format yymmdd)
    match = re.search(r'(\d{6})', os.path.basename(file))
    if not match:
        continue  # Skip files without a valid 6-digit date
    date_str = match.group(1)

    
    # Open the PDF with pdfplumber
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table is None:
                continue

            # Process each row in the table
            for row in table:
                if row is None or len(row) < 4:
                    continue
                # Skip header/footer rows based on keywords
                if ("Employee" in row[0]
                    or any("Page" in (cell or "") for cell in row)
                    or any("Grand" in (cell or "") for cell in row)):
                    continue

                employee = row[0].strip()
                avg_pieces = row[1].strip() if row[1] else ""
                avg_dollar = row[2].strip() if row[2] else ""
                avg_skus = row[3].strip() if row[3] else ""
                if not employee:
                    continue

                all_rows.append({
                    "Employee": employee,
                    "Date": date_str,
                    "Avg_Pieces_Hr": avg_pieces,
                    "Avg_$Hr": avg_dollar,
                    "Avg_Skus_Hr": avg_skus
                })

# Create a DataFrame from the extracted data
df = pd.DataFrame(all_rows)

if df.empty:
    raise ValueError("No data was extracted from the PDFs. Check your PDF structure or extraction logic.")

# Ensure expected columns exist
expected_cols = ["Avg_Pieces_Hr", "Avg_$Hr", "Avg_Skus_Hr"]
for col in expected_cols:
    if col not in df.columns:
        raise KeyError(f"Expected column '{col}' not found in the extracted data.")

# Convert numeric columns
df["Avg_Pieces_Hr"] = pd.to_numeric(df["Avg_Pieces_Hr"], errors="coerce")
df["Avg_$Hr"] = pd.to_numeric(df["Avg_$Hr"], errors="coerce")
df["Avg_Skus_Hr"] = pd.to_numeric(df["Avg_Skus_Hr"], errors="coerce")

# Convert the Date column to datetime (assume yymmdd format)
df["Date"] = pd.to_datetime(df["Date"], format="%y%m%d")

# ---------------------------
# STEP 2: Compute Improvements & Sort (Using Avg $/Hr)
# ---------------------------
def compute_improvement_dollar(group):
    group = group.sort_values("Date")
    first = group.iloc[0]["Avg_$Hr"]
    last = group.iloc[-1]["Avg_$Hr"]
    if pd.isna(first) or pd.isna(last):
        return None
    return last - first

improvements = df.groupby("Employee").apply(compute_improvement_dollar)
improvement_df = improvements.reset_index().rename(columns={0: "Improvement_$Hr"})

# Sort employees by improvement in dollars per hour (most improved first)
sorted_employees = improvement_df.sort_values("Improvement_$Hr", ascending=False)["Employee"].tolist()

# ---------------------------
# STEP 3: Create Separate PDF Report (Based on Avg $/Hr)
# ---------------------------
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# -- SUMMARY PAGE --
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Sorted Employee Improvement Summary ($/Hr)", ln=True, align="C")
pdf.ln(10)

# Summary table header
pdf.set_font("Arial", "B", 10)
col_width = pdf.w / 4 - 5
pdf.cell(col_width, 10, "Employee", border=1)
pdf.cell(col_width, 10, "Improvement", border=1)
pdf.cell(col_width, 10, "Latest Avg $/Hr", border=1)
pdf.cell(col_width, 10, "Earliest Avg $/Hr", border=1)
pdf.ln()

pdf.set_font("Arial", "", 10)
for employee in sorted_employees:
    emp_data = df[df["Employee"] == employee].sort_values("Date")
    if emp_data.empty:
        continue
    first = emp_data.iloc[0]["Avg_$Hr"]
    last = emp_data.iloc[-1]["Avg_$Hr"]
    improvement = last - first if pd.notna(first) and pd.notna(last) else None
    pdf.cell(col_width, 10, employee, border=1)
    pdf.cell(col_width, 10, f"{improvement:.2f}" if improvement is not None else "N/A", border=1)
    pdf.cell(col_width, 10, f"{last:.2f}" if pd.notna(last) else "N/A", border=1)
    pdf.cell(col_width, 10, f"{first:.2f}" if pd.notna(first) else "N/A", border=1)
    pdf.ln()

# -- DETAILED PAGES FOR EACH EMPLOYEE --
for employee in sorted_employees:
    emp_data = df[df["Employee"] == employee].sort_values("Date")
    if emp_data.empty:
        continue
    
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Employee: {employee}", ln=True, align="C")
    
    imp = compute_improvement_dollar(emp_data)
    pdf.set_font("Arial", "", 12)
    if imp is not None:
        pdf.cell(0, 10, f"Improvement in Avg $/Hr: {imp:.2f}", ln=True, align="C")
    else:
        pdf.cell(0, 10, "Improvement: N/A", ln=True, align="C")
    pdf.ln(5)
    
    # Table header for detailed page
    pdf.set_font("Arial", "B", 10)
    col_width_detail = pdf.w / 4 - 5
    pdf.cell(col_width_detail, 10, "Date", border=1)
    pdf.cell(col_width_detail, 10, "Avg $/Hr", border=1)
    pdf.cell(col_width_detail, 10, "Avg Pieces/Hr", border=1)
    pdf.cell(col_width_detail, 10, "Avg Skus/Hr", border=1)
    pdf.ln()
    
    # Table rows for detailed page
    pdf.set_font("Arial", "", 10)
    for _, row in emp_data.iterrows():
        pdf.cell(col_width_detail, 10, row["Date"].strftime("%Y-%m-%d"), border=1)
        pdf.cell(col_width_detail, 10, f"{row['Avg_$Hr']:.2f}" if pd.notna(row["Avg_$Hr"]) else "N/A", border=1)
        pdf.cell(col_width_detail, 10, f"{row['Avg_Pieces_Hr']:.2f}" if pd.notna(row["Avg_Pieces_Hr"]) else "N/A", border=1)
        pdf.cell(col_width_detail, 10, f"{row['Avg_Skus_Hr']:.2f}" if pd.notna(row["Avg_Skus_Hr"]) else "N/A", border=1)
        pdf.ln()
    
    # Generate a chart for Avg $/Hr over time
    emp_data = emp_data.sort_values("Date")
    plt.figure()
    plt.plot(emp_data["Date"], emp_data["Avg_$Hr"], marker="o")
    plt.title(f"{employee} - Avg $/Hr Over Time")
    plt.xlabel("Date")
    plt.ylabel("Avg $/Hr")
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    global_min = df["Avg_$Hr"].min()
    global_max = df["Avg_$Hr"].max()
    plt.ylim(global_min, global_max)
    plt.tight_layout()

    chart_filename = f"{employee}_dollar_chart.png"
    plt.savefig(chart_filename, bbox_inches="tight")
    plt.close()
    
    pdf.ln(5)
    pdf.image(chart_filename, x=10, y=pdf.get_y(), w=pdf.w - 20)
    os.remove(chart_filename)

# ---------------------------
# Scatter Plot: Initial vs Final Performance (Avg $/Hr)
# ---------------------------
# Compute the most recent overall average from the most recent PDF date based on Avg $/Hr
most_recent_date = df["Date"].max()
overall_recent_avg = df[df["Date"] == most_recent_date]["Avg_$Hr"].mean()

# Compute initial and final performance for each employee (using Avg $/Hr)
initials = []
finals = []
employee_names = []

for employee in sorted_employees:
    emp_data = df[df["Employee"] == employee].sort_values("Date")
    if emp_data.empty:
        continue
    initial = emp_data.iloc[0]["Avg_$Hr"]
    final = emp_data.iloc[-1]["Avg_$Hr"]
    initials.append(initial)
    finals.append(final)
    employee_names.append(employee)

plt.figure()

# Identify special employees that meet the condition:
# (initial < overall_recent_avg) and (final < initial)
special_indices = [i for i, (init, fin) in enumerate(zip(initials, finals))
                   if (init < overall_recent_avg) and (fin < init)]
num_special = len(special_indices)
cmap = plt.cm.get_cmap('tab10', num_special) if num_special > 0 else None

special_mapping = {}
special_counter = 1
texts = []
for i, (initial, final, employee) in enumerate(zip(initials, finals, employee_names)):
    if i in special_indices:
        color = cmap(special_counter - 1)
        plt.scatter(initial, final, color=color, zorder=3)
        plt.annotate(str(special_counter), (initial, final), textcoords="offset points",
                     xytext=(5, 5), ha='left', fontsize=9, fontweight='bold')
        special_mapping[special_counter] = employee
        special_counter += 1
    elif final < overall_recent_avg:
        plt.scatter(initial, final, color="orange", zorder=3)
        if final < initial:
            plt.scatter(initial, final, color="red", zorder=3)
    else:
        plt.scatter(initial, final, color="gray", zorder=3)

adjust_text(
    texts,
    arrowprops=dict(arrowstyle='->', color='black', shrinkA=5, shrinkB=5),
    expand_points=(513.2, 513.2),
    expand_text=(513.2, 513.2),
    force_points=(339.5, 339.5),
    force_text=(339.5, 339.5),
    lim=100
)

min_val = min(initials + finals)
max_val = max(initials + finals)
plt.plot([min_val, max_val], [min_val, max_val], 'r--', zorder=2, label="No Improvement")

plt.xlabel("Initial Avg $/Hr")
plt.ylabel("Final Avg $/Hr")
plt.title("Scatter Plot: Initial vs Final Performance ($/Hr)")
plt.grid(True, zorder=0)

import matplotlib.patches as mpatches
legend_handles = []
for number, employee in special_mapping.items():
    color = cmap(number - 1)
    patch = mpatches.Patch(color=color, label=f"{number}: {employee}")
    legend_handles.append(patch)
plt.legend(
    handles=legend_handles,
    title="Selected Employees",
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0.
)

chart_filename = "Initial_vs_Final_dollar_chart.png"
plt.savefig(chart_filename, bbox_inches="tight")
plt.close()

pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Scatter Plot: Initial vs Final Performance ($/Hr)", ln=True, align="C")
pdf.ln(10)
pdf.image(chart_filename, x=10, y=pdf.get_y(), w=pdf.w - 20)
os.remove(chart_filename)

# Save the final PDF report
output_filename = "Sorted_Employee_Performance_Report_Dollar.pdf"
pdf.output(output_filename)

print("Sorted employee performance report (based on Avg $/Hr) generated successfully as:", output_filename)
