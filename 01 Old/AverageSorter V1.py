# import re
# import os
# import glob
# import pdfplumber
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates  # <-- Added for improved date handling
# from fpdf import FPDF
# from adjustText import adjust_text

# # ---------------------------
# # STEP 1: Data Extraction
# # ---------------------------
# # Set the directory where your production PDF files are stored
# pdf_dir = r"C:\Users\Laptop 122\Desktop\Store Prep\06 Employee Reports\01 Old"  # update this to your folder with PDFs
# pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

# all_rows = []

# for file in pdf_files:
#     # Extract the 6-digit date from the filename (format yymmdd)
#     match = re.search(r'(\d{6})', os.path.basename(file))
#     if not match:
#         continue  # Skip files without a valid 6-digit date
#     date_str = match.group(1)

    
#     # Open the PDF with pdfplumber
#     with pdfplumber.open(file) as pdf:
#         for page in pdf.pages:
#             table = page.extract_table()
#             if table is None:
#                 continue

#             # Process each row in the table
#             for row in table:
#                 if row is None or len(row) < 4:
#                     continue
#                 # Skip header/footer rows based on keywords
#                 if ("Employee" in row[0]
#                     or any("Page" in (cell or "") for cell in row)
#                     or any("Grand" in (cell or "") for cell in row)):
#                     continue

#                 employee = row[0].strip()
#                 avg_pieces = row[1].strip() if row[1] else ""
#                 avg_dollar = row[2].strip() if row[2] else ""
#                 avg_skus = row[3].strip() if row[3] else ""
#                 if not employee:
#                     continue

#                 all_rows.append({
#                     "Employee": employee,
#                     "Date": date_str,
#                     "Avg_Pieces_Hr": avg_pieces,
#                     "Avg_$Hr": avg_dollar,
#                     "Avg_Skus_Hr": avg_skus
#                 })

# # Create a DataFrame from the extracted data
# df = pd.DataFrame(all_rows)

# if df.empty:
#     raise ValueError("No data was extracted from the PDFs. Check your PDF structure or extraction logic.")

# # Ensure expected columns exist
# expected_cols = ["Avg_Pieces_Hr", "Avg_$Hr", "Avg_Skus_Hr"]
# for col in expected_cols:
#     if col not in df.columns:
#         raise KeyError(f"Expected column '{col}' not found in the extracted data.")

# # Convert numeric columns
# df["Avg_Pieces_Hr"] = pd.to_numeric(df["Avg_Pieces_Hr"], errors="coerce")
# df["Avg_$Hr"] = pd.to_numeric(df["Avg_$Hr"], errors="coerce")
# df["Avg_Skus_Hr"] = pd.to_numeric(df["Avg_Skus_Hr"], errors="coerce")

# # Convert the Date column to datetime (assume yymmdd => 2025-xx-xx, etc.)
# df["Date"] = pd.to_datetime(df["Date"], format="%y%m%d")

# # ---------------------------
# # STEP 2: Compute Improvements & Sort
# # ---------------------------
# def compute_improvement(group):
#     group = group.sort_values("Date")
#     first = group.iloc[0]["Avg_Pieces_Hr"]
#     last = group.iloc[-1]["Avg_Pieces_Hr"]
#     if pd.isna(first) or pd.isna(last):
#         return None
#     return last - first

# improvements = df.groupby("Employee").apply(compute_improvement)
# improvement_df = improvements.reset_index().rename(columns={0: "Improvement_Pieces_Hr"})

# # Sort employees by improvement (most improved first)
# sorted_employees = improvement_df.sort_values("Improvement_Pieces_Hr", ascending=False)["Employee"].tolist()

# # ---------------------------
# # STEP 3: Create Single PDF Report
# # ---------------------------
# pdf = FPDF()
# pdf.set_auto_page_break(auto=True, margin=15)

# # -- SUMMARY PAGE --
# pdf.add_page()
# pdf.set_font("Arial", "B", 16)
# pdf.cell(0, 10, "Sorted Employee Improvement Summary", ln=True, align="C")
# pdf.ln(10)

# # Summary table header
# pdf.set_font("Arial", "B", 10)
# col_width = pdf.w / 4 - 5
# pdf.cell(col_width, 10, "Employee", border=1)
# pdf.cell(col_width, 10, "Improvement", border=1)
# pdf.cell(col_width, 10, "Latest Avg Pieces", border=1)
# pdf.cell(col_width, 10, "Earliest Avg Pieces", border=1)
# pdf.ln()

# pdf.set_font("Arial", "", 10)
# for employee in sorted_employees:
#     emp_data = df[df["Employee"] == employee].sort_values("Date")
#     if emp_data.empty:
#         continue
#     first = emp_data.iloc[0]["Avg_Pieces_Hr"]
#     last = emp_data.iloc[-1]["Avg_Pieces_Hr"]
#     improvement = last - first if pd.notna(first) and pd.notna(last) else None
#     pdf.cell(col_width, 10, employee, border=1)
#     pdf.cell(col_width, 10, f"{improvement:.2f}" if improvement is not None else "N/A", border=1)
#     pdf.cell(col_width, 10, f"{last:.2f}" if pd.notna(last) else "N/A", border=1)
#     pdf.cell(col_width, 10, f"{first:.2f}" if pd.notna(first) else "N/A", border=1)
#     pdf.ln()

# # -- DETAILED PAGES FOR EACH EMPLOYEE --
# for employee in sorted_employees:
#     emp_data = df[df["Employee"] == employee].sort_values("Date")
#     if emp_data.empty:
#         continue
    
#     pdf.add_page()
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, f"Employee: {employee}", ln=True, align="C")
    
#     imp = compute_improvement(emp_data)
#     pdf.set_font("Arial", "", 12)
#     if imp is not None:
#         pdf.cell(0, 10, f"Improvement in Avg Pieces/Hr: {imp:.2f}", ln=True, align="C")
#     else:
#         pdf.cell(0, 10, "Improvement: N/A", ln=True, align="C")
#     pdf.ln(5)
    
#     # Table header
#     pdf.set_font("Arial", "B", 10)
#     col_width_detail = pdf.w / 4 - 5
#     pdf.cell(col_width_detail, 10, "Date", border=1)
#     pdf.cell(col_width_detail, 10, "Avg Pieces/Hr", border=1)
#     pdf.cell(col_width_detail, 10, "Avg $/Hr", border=1)
#     pdf.cell(col_width_detail, 10, "Avg Skus/Hr", border=1)
#     pdf.ln()
    
#     # Table rows
#     pdf.set_font("Arial", "", 10)
#     for _, row in emp_data.iterrows():
#         pdf.cell(col_width_detail, 10, row["Date"].strftime("%Y-%m-%d"), border=1)
#         pdf.cell(col_width_detail, 10, f"{row['Avg_Pieces_Hr']:.2f}" if pd.notna(row["Avg_Pieces_Hr"]) else "N/A", border=1)
#         pdf.cell(col_width_detail, 10, f"{row['Avg_$Hr']:.2f}" if pd.notna(row["Avg_$Hr"]) else "N/A", border=1)
#         pdf.cell(col_width_detail, 10, f"{row['Avg_Skus_Hr']:.2f}" if pd.notna(row["Avg_Skus_Hr"]) else "N/A", border=1)
#         pdf.ln()
    
#     # Generate a chart with improved date formatting
#     plt.figure()
#     # Sort data again just to be safe
#     emp_data = emp_data.sort_values("Date")
#     plt.plot(emp_data["Date"], emp_data["Avg_Pieces_Hr"], marker="o")

#     # Calculate global min and max for Avg_Pieces_Hr (do this once, outside the loop)
#     global_min = df["Avg_Pieces_Hr"].min()
#     global_max = df["Avg_Pieces_Hr"].max()


#     # Inside the loop for each employee's chart:
#     plt.figure()
#     emp_data = emp_data.sort_values("Date")
#     plt.plot(emp_data["Date"], emp_data["Avg_Pieces_Hr"], marker="o")

#     plt.title(f"{employee} - Avg Pieces/Hr Over Time")
#     plt.xlabel("Date")
#     plt.ylabel("Avg Pieces/Hr")
#     plt.grid(True)

#     # Improved date formatting
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto chooses daily/monthly
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format labels
#     plt.xticks(rotation=45)
#     plt.ylim(global_min, global_max)
#     plt.tight_layout()

#     chart_filename = f"{employee}_chart.png"
#     plt.savefig(chart_filename, bbox_inches="tight")
#     plt.close()
    
#     pdf.ln(5)
#     pdf.image(chart_filename, x=10, y=pdf.get_y(), w=pdf.w - 20)
#     os.remove(chart_filename)


# # ---------------------------
# # Scatter Plot: Initial vs Final Performance with Unique Colors for Selected Employees
# # ---------------------------
# # Compute the most recent overall average from the most recent PDF date
# most_recent_date = df["Date"].max()
# overall_recent_avg = df[df["Date"] == most_recent_date]["Avg_Pieces_Hr"].mean()

# # Compute initial and final performance for each employee
# initials = []
# finals = []
# employee_names = []

# for employee in sorted_employees:
#     emp_data = df[df["Employee"] == employee].sort_values("Date")
#     if emp_data.empty:
#         continue
#     initial = emp_data.iloc[0]["Avg_Pieces_Hr"]
#     final = emp_data.iloc[-1]["Avg_Pieces_Hr"]
#     initials.append(initial)
#     finals.append(final)
#     employee_names.append(employee)

# plt.figure()

# # Identify special employees that meet the condition:
# # (initial < overall_recent_avg) and (final < initial)
# special_indices = [i for i, (init, fin) in enumerate(zip(initials, finals))
#                    if (init < overall_recent_avg) and (fin < init)]
# num_special = len(special_indices)
# cmap = plt.cm.get_cmap('tab10', num_special) if num_special > 0 else None

# # Dictionary to map a unique number label to employee names
# special_mapping = {}
# special_counter = 1
# texts = []
# # Plot each employee’s point
# for i, (initial, final, employee) in enumerate(zip(initials, finals, employee_names)):
#     if i in special_indices:
#         # Assign a unique color from the colormap
#         color = cmap(special_counter - 1)
#         plt.scatter(initial, final, color=color, zorder=3)
#         # Annotate with a number label
#         plt.annotate(str(special_counter), (initial, final), textcoords="offset points",
#                      xytext=(5, 5), ha='left', fontsize=9, fontweight='bold')
#         special_mapping[special_counter] = employee
#         special_counter += 1
#     elif final < overall_recent_avg:
#         plt.scatter(initial, final, color="orange", zorder=3)
#         if final < initial:
#             plt.scatter(initial, final, color="red", zorder=3)
#             # Store text object in a list
#             # t = plt.text(initial, final, employee, fontsize=7)
#             # texts.append(t)
#     else:
#         # Plot all other points in a default color
#         plt.scatter(initial, final, color="gray", zorder=3)

# # Now call adjust_text once, after plotting all points:
# adjust_text(
#     texts,
#     arrowprops=dict(arrowstyle='->', color='black', shrinkA=5, shrinkB=5),  
#     expand_points=(513.2, 513.2),
#     expand_text=(513.2, 513.2),
#     force_points=(339.5, 339.5),
#     force_text=(339.5, 339.5),
#     lim=100  # max it1erations
# )

# # Add a reference line (y=x) indicating no improvement
# min_val = min(initials + finals)
# max_val = max(initials + finals)
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', zorder=2, label="No Improvement")

# plt.xlabel("Initial Avg Pieces/Hr")
# plt.ylabel("Final Avg Pieces/Hr")
# plt.title("Scatter Plot: Initial vs Final Performance")
# plt.grid(True, zorder=0)

# # Build a legend (key) for the special employees
# import matplotlib.patches as mpatches
# legend_handles = []
# for number, employee in special_mapping.items():
#     color = cmap(number - 1)
#     patch = mpatches.Patch(color=color, label=f"{number}: {employee}")
#     legend_handles.append(patch)
# # plt.legend(handles=legend_handles, title="Selected Employees", loc="best")
# plt.legend(
#     handles=legend_handles,
#     title="Selected Employees",
#     loc="upper left",
#     bbox_to_anchor=(1.05, 1),  # shift the legend to the right
#     borderaxespad=0.
# )


# # Save and add the scatter plot to the PDF report
# chart_filename = "Initial_vs_Final_chart.png"
# plt.savefig(chart_filename, bbox_inches="tight")
# plt.close()

# pdf.add_page()
# pdf.set_font("Arial", "B", 16)
# pdf.cell(0, 10, "Scatter Plot: Initial vs Final Performance", ln=True, align="C")
# pdf.ln(10)
# pdf.image(chart_filename, x=10, y=pdf.get_y(), w=pdf.w - 20)
# os.remove(chart_filename)





# # Save the single combined PDF report
# output_filename = "Sorted_Employee_Performance_Report.pdf"
# pdf.output(output_filename)

# print("Sorted employee performance report generated successfully as:", output_filename)



import re
import os
import glob
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # <-- Added for improved date handling
from fpdf import FPDF
from adjustText import adjust_text
from PIL import Image  # Added to measure chart size

# ---------------------------
# STEP 1: Data Extraction
# ---------------------------
pdf_dir = r"C:\Users\Laptop 122\Desktop\Store Prep\06 Employee Reports\01 Old"  # update path
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

all_rows = []
for file in pdf_files:
    match = re.search(r'(\d{6})', os.path.basename(file))
    if not match:
        continue
    date_str = match.group(1)
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table is None:
                continue
            for row in table:
                if not row or len(row) < 4:
                    continue
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

# Create DataFrame
df = pd.DataFrame(all_rows)
if df.empty:
    raise ValueError("No data was extracted from the PDFs. Check your PDF structure or extraction logic.")
# Convert types
for col in ["Avg_Pieces_Hr", "Avg_$Hr", "Avg_Skus_Hr"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Date"] = pd.to_datetime(df["Date"], format="%y%m%d")

# Compute improvements
def compute_improvement(group):
    group = group.sort_values("Date")
    first = group.iloc[0]["Avg_Pieces_Hr"]
    last = group.iloc[-1]["Avg_Pieces_Hr"]
    if pd.isna(first) or pd.isna(last):
        return None
    return last - first

improvements = df.groupby("Employee").apply(compute_improvement)
improvement_df = improvements.reset_index().rename(columns={0: "Improvement_Pieces_Hr"})
sorted_employees = improvement_df.sort_values("Improvement_Pieces_Hr", ascending=False)["Employee"].tolist()

# Create PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Summary page
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Sorted Employee Improvement Summary", ln=True, align="C")
pdf.ln(10)
col_width = pdf.w / 4 - 5
pdf.set_font("Arial", "B", 10)
for header in ["Employee", "Improvement", "Latest Avg Pieces", "Earliest Avg Pieces"]:
    pdf.cell(col_width, 10, header, border=1)
pdf.ln()
pdf.set_font("Arial", "", 10)
for emp in sorted_employees:
    emp_data = df[df["Employee"] == emp].sort_values("Date")
    if emp_data.empty:
        continue
    first = emp_data.iloc[0]["Avg_Pieces_Hr"]
    last = emp_data.iloc[-1]["Avg_Pieces_Hr"]
    improvement = last - first if pd.notna(first) and pd.notna(last) else None
    pdf.cell(col_width, 10, emp, border=1)
    pdf.cell(col_width, 10, f"{improvement:.2f}" if improvement is not None else "N/A", border=1)
    pdf.cell(col_width, 10, f"{last:.2f}" if pd.notna(last) else "N/A", border=1)
    pdf.cell(col_width, 10, f"{first:.2f}" if pd.notna(first) else "N/A", border=1)
    pdf.ln()

# Detailed pages
for employee in sorted_employees:
    emp_data = df[df["Employee"] == employee].sort_values("Date")
    if emp_data.empty:
        continue
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Employee: {employee}", ln=True, align="C")
    imp = compute_improvement(emp_data)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Improvement in Avg Pieces/Hr: {imp:.2f}" if imp is not None else "Improvement: N/A", ln=True, align="C")
    pdf.ln(5)
    # Table
    pdf.set_font("Arial", "B", 10)
    col_w = pdf.w / 4 - 5
    for h in ["Date", "Avg Pieces/Hr", "Avg $/Hr", "Avg Skus/Hr"]:
        pdf.cell(col_w, 10, h, border=1)
    pdf.ln()
    pdf.set_font("Arial", "", 10)
    for _, row in emp_data.iterrows():
        pdf.cell(col_w, 10, row["Date"].strftime("%Y-%m-%d"), border=1)
        pdf.cell(col_w, 10, f"{row['Avg_Pieces_Hr']:.2f}" if pd.notna(row['Avg_Pieces_Hr']) else "N/A", border=1)
        pdf.cell(col_w, 10, f"{row['Avg_$Hr']:.2f}" if pd.notna(row['Avg_$Hr']) else "N/A", border=1)
        pdf.cell(col_w, 10, f"{row['Avg_Skus_Hr']:.2f}" if pd.notna(row['Avg_Skus_Hr']) else "N/A", border=1)
        pdf.ln()
    # Plot
    plt.figure()
    plt.plot(emp_data["Date"], emp_data["Avg_Pieces_Hr"], marker="o")
    plt.title(f"{employee} - Avg Pieces/Hr Over Time")
    plt.xlabel("Date")
    plt.ylabel("Avg Pieces/Hr")
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    global_min = df["Avg_Pieces_Hr"].min()
    global_max = df["Avg_Pieces_Hr"].max()
    plt.ylim(global_min, global_max)
    plt.tight_layout()
    chart_filename = f"{employee}_chart.png"
    plt.savefig(chart_filename, bbox_inches="tight")
    plt.close()
    pdf.ln(5)
    # Ensure there's enough room and close image file before removal
    with Image.open(chart_filename) as img:
        img_w, img_h = img.size
    chart_w = pdf.w - 20
    chart_h = img_h * chart_w / img_w
    if pdf.get_y() + chart_h > pdf.h - pdf.b_margin:
        pdf.add_page()
    pdf.image(chart_filename, x=10, y=pdf.get_y(), w=chart_w)
    os.remove(chart_filename)

# Scatter plot page
most_recent_date = df["Date"].max()
overall_recent_avg = df[df["Date"] == most_recent_date]["Avg_Pieces_Hr"].mean()
initials, finals, names = [], [], []
for emp in sorted_employees:
    ed = df[df["Employee"] == emp].sort_values("Date")
    if ed.empty: continue
    initials.append(ed.iloc[0]["Avg_Pieces_Hr"])
    finals.append(ed.iloc[-1]["Avg_Pieces_Hr"])
    names.append(emp)

plt.figure()
special = [idx for idx,(i0,f) in enumerate(zip(initials, finals)) if i0<overall_recent_avg and f<i0]
cmap = plt.cm.get_cmap('tab10', len(special)) if special else None
mapping, counter, texts = {}, 1, []
for idx,(i0,f,emp) in enumerate(zip(initials, finals, names)):
    if idx in special:
        color = cmap(counter-1)
        plt.scatter(i0, f, color=color, zorder=3)
        plt.annotate(str(counter), (i0, f), textcoords="offset points", xytext=(5,5), ha='left', fontsize=9, fontweight='bold')
        mapping[counter] = emp
        counter+=1
    elif f<overall_recent_avg:
        plt.scatter(i0, f, color="orange", zorder=3)
        if f<i0:
            plt.scatter(i0, f, color="red", zorder=3)
    else:
        plt.scatter(i0, f, color="gray", zorder=3)
adjust_text(texts, arrowprops=dict(arrowstyle='->',color='black',shrinkA=5,shrinkB=5),expand_points=(513.2,513.2),expand_text=(513.2,513.2),force_points=(339.5,339.5),force_text=(339.5,339.5),lim=100)
minv, maxv = min(initials+finals), max(initials+finals)
plt.plot([minv,maxv],[minv,maxv],'r--',zorder=2,label="No Improvement")
plt.xlabel("Initial Avg Pieces/Hr")
plt.ylabel("Final Avg Pieces/Hr")
plt.title("Scatter Plot: Initial vs Final Performance")
plt.grid(True, zorder=0)
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=cmap(n-1), label=f"{n}: {e}") for n,e in mapping.items()]
plt.legend(handles=handles, title="Selected Employees", loc="upper left", bbox_to_anchor=(1.05,1), borderaxespad=0.)
chart_filename = "Initial_vs_Final_chart.png"
plt.savefig(chart_filename, bbox_inches="tight")
plt.close()
pdf.add_page()
pdf.set_font("Arial","B",16)
pdf.cell(0,10,"Scatter Plot: Initial vs Final Performance",ln=True,align="C")
pdf.ln(10)
pdf.image(chart_filename, x=10, y=pdf.get_y(), w=pdf.w-20)
os.remove(chart_filename)

# Save report
output_filename = "Sorted_Employee_Performance_Report.pdf"
pdf.output(output_filename)
print("Report generated:", output_filename)
