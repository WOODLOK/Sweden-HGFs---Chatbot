import os
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import openai

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from st_aggrid import AgGrid, GridOptionsBuilder

import re
from collections import Counter, defaultdict
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from sklearn.metrics.pairwise import cosine_similarity

# === Load API key ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY not found.")
client = openai.OpenAI(api_key=api_key)

st.markdown("""
<style>
:root {
    --whu-blue: #1A3975;
    --whu-light-blue: #00A5DC;
    --kellogg-purple: #4F2683;
    --silver-grey: #E6EBF0;
    --anthracite: #505459;
    --black: #000000;
    --dark-petrol: #006E78;
    --petrol: #00AAB9;
    --orange: #EB5A0A;
    --yellow: #FFE600;
    --green: #1CAD4B;
    --chart-80: #338B93;
    --chart-60: #66A8AE;
    --chart-40: #99C0C9;
    --chart-20: #CCE2E4;
}

/* Main background */
[data-testid="stAppViewContainer"] {
    background: #e6ebf0; /* fallback */
    background: linear-gradient(180deg, #e6ebf0 0%, #f4f8fc 100%);
}

/* Hide the top bar */
header[data-testid="stHeader"] {
    background: transparent !important;
}

header[data-testid="stHeader"] * {
    color: var(--whu-blue) !important;
}

/* Top bar deploy button styling */
header[data-testid="stHeader"] button[kind="primary"] {
    background-color: var(--whu-blue) !important;
    color: #fff !important;
}

header[data-testid="stHeader"] [data-testid="stToolbar"] button {
    color: var(--whu-blue) !important;
}

/* Remove top padding from main container to compensate for removed header */
.main > .block-container {
    padding-top: 1rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #1A3975;
    border-right: 2px solid var(--whu-blue);
    box-shadow: 2px 0 12px 0 rgba(26,57,117,0.10), 0 4px 12px 0 rgba(0,0,0,0.04);
    z-index: 2;
}

/* Sidebar headers - consistent but smaller font size */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2 {
    font-size: 20px !important;
    font-weight: 600 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4 {
    font-size: 16px !important;
    font-weight: 600 !important;
    margin-bottom: 0.75rem !important;
}

[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] li {
    font-size: 14px !important;
    margin-bottom: 0.5rem !important;
}

/* Standardize section spacing in sidebar */
[data-testid="stSidebar"] .block-container > div:first-child {
    padding-top: 1rem !important;
}

/* First title (no top margin) */
[data-testid="stSidebar"] .block-container > div:first-child h1:first-child {
    margin-top: 0 !important;
    font-size: 22px !important;
}

[data-testid="stSidebar"] h2 {
    margin-top: 2rem !important;
}

[data-testid="stMarkdown"] {
    margin-bottom: 1rem !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] p, [data-testid="stSidebar"] li, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
    color: #fff !important;
}

/* Sidebar buttons (New Chat/Chat History) */
[data-testid="stSidebar"] button, [data-testid="stSidebar"] .stButton>button {
    background: transparent !important;
    color: #fff !important;
    border: 2px solid #fff !important;
    border-radius: 10px !important;
    font-weight: 600;
    box-shadow: 0 2px 8px 0 rgba(26,57,117,0.10);
    margin-bottom: 0.3em !important;
    padding: 0.4em 1em !important;
    transition: all 0.2s ease;
    width: 100%;
    text-align: left;
}

[data-testid="stSidebar"] button:hover, [data-testid="stSidebar"] .stButton>button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #fff !important;
    border: 2px solid var(--whu-light-blue) !important;
}

/* Chat bubbles */
section[data-testid="stChatMessage"] {
    border-radius: 18px;
    margin-bottom: 1.2em;
    box-shadow: 0 2px 8px 0 rgba(32,78,151,0.07);
    background: #eaf2fa !important; /* lighter blue for chat area */
    border: 1.5px solid var(--whu-light-blue);
}
section[data-testid="stChatMessage"] .stMarkdown {
    color: var(--anthracite);
    font-size: 1.08rem;
}

/* User chat bubble */
section[data-testid="stChatMessage"]:has(.stMarkdown[data-testid="stMarkdownContainer"]) {
    background: #d6e6f7 !important; /* even lighter for user bubble */
    color: #204E97;
    border: 1.5px solid var(--whu-blue);
}

/* Input box styling */
section[data-testid="stChatInput"] textarea {
    border-radius: 12px;
    border: 2px solid var(--whu-blue);
    background: #fff;
    font-size: 1.08rem;
    box-shadow: 0 2px 8px 0 rgba(32,78,151,0.08);
    transition: border 0.2s;
}
section[data-testid="stChatInput"] textarea:focus {
    border: 2px solid var(--whu-light-blue);
    outline: none;
}

/* Button styling */
button[kind="primary"], .stButton>button {
    background: var(--whu-blue);
    color: #fff;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    box-shadow: 0 2px 8px 0 rgba(32,78,151,0.10);
    transition: background 0.2s, box-shadow 0.2s;
}
button[kind="primary"]:hover, .stButton>button:hover {
    background: var(--whu-light-blue);
    color: #fff;
    box-shadow: 0 4px 16px 0 rgba(0,165,220,0.15);
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: var(--whu-blue);
    color: #fff;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    box-shadow: 0 2px 8px 0 rgba(32,78,151,0.10);
    transition: background 0.2s, box-shadow 0.2s;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--whu-light-blue);
    color: #fff;
    box-shadow: 0 4px 16px 0 rgba(0,165,220,0.15);
}

/* Table and AgGrid tweaks */
.ag-theme-streamlit .ag-header, .ag-theme-streamlit .ag-root-wrapper {
    background: var(--silver-grey);
    color: var(--whu-blue);
    border-radius: 10px;
}

/* Headings and links */
h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--whu-blue);
    font-weight: 700;
}
a, .stMarkdown a {
    color: var(--whu-light-blue);
    text-decoration: underline;
}

/* Miscellaneous */
.stAlert {
    border-left: 6px solid var(--whu-blue);
    background: var(--chart-20);
    color: var(--anthracite);
    border-radius: 10px;
}

/* Bullet points */
ul, ol {
    color: var(--anthracite);
}
ul li::marker {
    color: var(--whu-blue);
}
ul ul li::marker, ul ol li::marker {
    color: var(--chart-80);
}

/* Chat area background */
[data-testid="stAppViewContainer"] {
    background: #e6ebf0; /* fallback */
    background: linear-gradient(180deg, #e6ebf0 0%, #f4f8fc 100%);
}
section[data-testid="stChatMessage"] {
    background: #eaf2fa !important; /* lighter blue for chat area */
    border: 1.5px solid var(--whu-light-blue);
}
section[data-testid="stChatMessage"]:has(.stMarkdown[data-testid="stMarkdownContainer"]) {
    background: #d6e6f7 !important; /* even lighter for user bubble */
    color: #204E97;
    border: 1.5px solid var(--whu-blue);
}

/* Title and section spacing */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    font-size: 20px !important;
    font-weight: 600 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

/* Main title specific styling - sidebar */
[data-testid="stSidebar"] .block-container > div:first-child h1:first-child {
    margin-top: 0 !important;
    font-size: 22px !important;
}

/* Main chat area title styling */
.main .block-container::before {
    content: "Competitor Chatbot for Sweden Consistent HGFs";
    display: block;
    font-size: 22px !important;
    font-weight: 600;
    color: var(--whu-blue);
    margin: 0.5rem 0 1.5rem 0;
    padding-left: 1rem;
}

/* Adjust main content area padding */
.main .block-container {
    padding-top: 1rem !important;
}

/* First title (no top margin) */
[data-testid="stSidebar"] .block-container > div:first-child h1:first-child {
    margin-top: 0 !important;
}

/* Ensure consistent markdown spacing */
[data-testid="stMarkdown"] {
    margin-bottom: 1rem !important;
}

/* Chat History section spacing */
[data-testid="stSidebar"] div[data-testid="stExpander"] {
    margin-top: 1rem !important;
}

/* Reduce space between Chat History and All Sessions */
[data-testid="stSidebar"] div[data-testid="stExpander"] > div:first-child {
    margin-bottom: 0.5rem !important;
}

/* Remove extra margins from markdown elements */
[data-testid="stMarkdown"] p {
    margin-bottom: 0.25rem !important;
}

[data-testid="stMarkdown"] hr {
    margin: 0.5rem 0 !important;
}

/* Chat history button spacing */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stButton>button {
    margin-bottom: 0.3em !important;
    padding: 0.4em 1em !important;
}

/* Main title styling */
.main-title {
    font-size: 24px !important;
    font-weight: 600 !important;
    color: #1A3975 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Sidebar title specific styling */
[data-testid="stSidebar"] .block-container > div:first-child h1:first-child {
    margin-top: 0 !important;
    font-size: 24px !important;
}

/* Adjust main content area spacing */
.main .block-container {
    padding-top: 0 !important;
}

/* Chat message styling */
[data-testid="stChatMessage"] {
    margin-top: 1rem !important;
}

/* Remove default margins from the chat container */
[data-testid="stChatMessageContainer"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Welcome message specific styling */
[data-testid="stChatMessage"]:first-child {
    margin-top: 1rem !important;
}

</style>
""", unsafe_allow_html=True)

# Add the main title at the top of the chat area

def gpt_generate_summary(top5_df, input_desc, filters=None):
    def extract_field_distributions(df):
        from collections import Counter
        def top_counts(series, sep=';'):
            counter = Counter()
            for val in series.dropna():
                labels = [x.strip() for x in val.split(sep)] if sep in val else [val.strip()]
                counter.update(labels)
            return counter.most_common(5)
        # Construct full sentence descriptions for each dimension:
        dist = {
            "Founded Year": f"The companies were founded between {df['Founded Year'].min()} and {df['Founded Year'].max()}.",
            "Number of employees 2023": f"The workforce size ranges from {df['Number of employees 2023'].min()} to {df['Number of employees 2023'].max()}.",
            "City Latin Alphabet": ("The companies are located in key cities. Top cities include: " +
                                      ", ".join([f"{k} ({v} companies)" for k, v in top_counts(df["City Latin Alphabet"])]) + "."),
            "Customer Segment": ("The customer segments vary, with examples such as: " +
                                 ", ".join([f"{k} ({v} companies)" for k, v in top_counts(df["Customer Segment"])]) + "."),
            "Growth Category": ("The companies belong to various growth categories, for example: " +
                                ", ".join([f"{k} ({v} companies)" for k, v in top_counts(df["Growth Category"])]) + "."),
            "BvD sectors": ("Key sectors in the dataset include: " +
                           ", ".join([f"{k} ({v} companies)" for k, v in top_counts(df["BvD sectors"])]) + "."),
            "All Topics": ("The main topics covered are: " +
                           ", ".join([f"{k} ({v} companies)" for k, v in top_counts(df["All Topics"])]) + ".")
        }
        # Added descriptions for Final Score and Company Age in a range format.
        if "Final Score" in df.columns:
            dist["Final Score"] = f"The top-5 matching scores range from {df['Final Score'].min():.2f} to {df['Final Score'].max():.2f}."
        if "Company Age" in df.columns:
            dist["Company Age"] = f"The company ages range from {df['Company Age'].min()} to {df['Company Age'].max()} years."
        return dist

    dist = extract_field_distributions(top5_df)
    filter_text = "No filters applied." if not filters else f"Filters used: {filters}"
    
    # Create full-sentence insights for the prompt.
    insight_lines = []
    for field, sentence in dist.items():
        insight_lines.append(f"- {field}: {sentence}")
    insight_block = "\n".join(insight_lines)
    
    prompt = f"""
You are a business insight analyst summarizing a comparison between one input company and its top-5 similar competitors.

Input company description:
\"\"\"{input_desc}\"\"\"

{filter_text}

Key Insights based on data distributions:
{insight_block}

Please return the following in plain text:
1. Provide a one-sentence summary of common patterns across the top-5 companies.
2. Provide one paragraph describing key differences and variations.
3. Provide one paragraph of strategic recommendations tailored to the input.
4. Provide one paragraph of caveats or limitations in the result.
Respond with one numbered item per line (1~4).
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a strategic business analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        import re
        result_lines = response.choices[0].message.content.strip().split("\n")
        # Remove any leading numbers (e.g., "1. ", "2. ") from each line.
        result_lines = [re.sub(r'^\s*\d+\.\s*', '', line) for line in result_lines]
        # If the caveats (fourth item) is empty or very simple, remove it.
        if len(result_lines) < 4 or not result_lines[3].strip() or result_lines[3].strip().lower() in ["n/a", ""]:
            result_lines = result_lines[:3]
        # Use the full sentence insights as field insights.
        field_insights = dist
        return result_lines, field_insights
    except Exception as e:
        print("GPT error:", e)
        # Fallback: generate complete sentences with numeric ranges if available.
        final_min = top5_df["Final Score"].min() if "Final Score" in top5_df.columns else None
        final_max = top5_df["Final Score"].max() if "Final Score" in top5_df.columns else None
        age_min = top5_df["Company Age"].min() if "Company Age" in top5_df.columns else None
        age_max = top5_df["Company Age"].max() if "Company Age" in top5_df.columns else None
        fallback_lines = []
        if final_min is not None and final_max is not None:
            fallback_lines.append(f"The top-5 matching scores range from {final_min:.2f} to {final_max:.2f}.")
        else:
            fallback_lines.append("Matching scores are available.")
        if age_min is not None and age_max is not None:
            fallback_lines.append(f"The company ages in the top-5 list range from {age_min} to {age_max} years.")
        else:
            fallback_lines.append("Company ages vary significantly.")
        fallback_lines.append("Strategic recommendation: Consider refining your strategy based on vertical differentiation and targeting underrepresented segments.")
        
        fallback_insights = {
            "Founded Year": f"The companies were founded between {top5_df['Founded Year'].min()} and {top5_df['Founded Year'].max()}.",
            "Number of employees 2023": f"The workforce size ranges from {top5_df['Number of employees 2023'].min()} to {top5_df['Number of employees 2023'].max()}.",
            "City Latin Alphabet": "The companies are primarily located in major economic hubs.",
            "Customer Segment": "Customer segments vary, with B2B being the most common.",
            "Growth Category": "The companies belong to various growth categories.",
            "BvD sectors": "Key sectors include wholesale, technology, and services.",
            "All Topics": "Main topics covered include logistics, automation, and SaaS.",
            "Final Score": f"The top-5 matching scores range from {final_min:.2f} to {final_max:.2f}." if final_min is not None else "",
            "Company Age": f"The company ages range from {age_min} to {age_max} years." if age_min is not None else ""
        }
        return fallback_lines, fallback_insights


def generate_report_pdf(comparison_df: pd.DataFrame, input_desc: str, filters: str = None) -> BytesIO:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from io import BytesIO
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import pandas as pd
    
    def draw_wrapped_text(c, text, x, y, max_width, font_name="Helvetica", font_size=12, leading=14):
        c.setFont(font_name, font_size)
        text_object = c.beginText()
        text_object.setTextOrigin(x, y)
        text_object.setLeading(leading)

        words = text.split()
        line = ""
        for word in words:
            test_line = line + word + " "
            if c.stringWidth(test_line, font_name, font_size) <= max_width:
                line = test_line
            else:
                text_object.textLine(line.strip())
                line = word + " "
        if line:
            text_object.textLine(line.strip())

        c.drawText(text_object)
        return text_object.getY()

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    # Determine the top-5 competitors dataframe.
    top5 = comparison_df.iloc[1:] if comparison_df.index[0] == "INPUT" else comparison_df.copy()
    summary_lines, field_insights = gpt_generate_summary(top5, input_desc, filters)
    input_is_present = comparison_df.index[0] == "INPUT"

    # === Executive Summary ===
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, H - 50, "Executive Summary")
    c.setFont("Helvetica", 12)
    y = H - 90

    intro = "This report is based on the input company description:" if len(input_desc.strip().split()) >= 4 else "This report is based on the user's input:"
    y = draw_wrapped_text(c, intro, 50, y, W - 100, font_size=12)
    y -= 10
    y = draw_wrapped_text(c, input_desc.strip(), 60, y, W - 120, font_name="Helvetica-Oblique", font_size=11)
    y -= 10

    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Top-5 matched competitors:")
    y -= 15
    for name in top5["Company name Latin alphabet"].tolist():
        c.drawString(60, y, f"• {name}")
        y -= 12

    y -= 10
    if filters:
        c.setFont("Helvetica", 10)
        y = draw_wrapped_text(c, f"Applied Filters: {filters}", 50, y, W - 100, font_size=10)
        y -= 10

    # Output the first two summary paragraphs (removing any numbering)
    for line in summary_lines[:2]:
        y = draw_wrapped_text(c, line.strip(), 50, y, W - 100)
        y -= 10

    c.showPage()

    # === Methodology ===
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, H - 50, "Methodology")
    c.setFont("Helvetica", 12)
    y = H - 90
    methodology_lines = [
        "• Data source: Sweden_final_filtered.csv + precomputed OpenAI text embeddings",
        "• Matching: cosine similarity over product/activity embeddings",
        "• Final Score = 0.9 * semantic similarity + 0.1 * company size (number of employees)",
        "• Threshold = 0.35 → Top‐5 results selected"
    ]
    for line in methodology_lines:
        y = draw_wrapped_text(c, line, 50, y, W - 100, font_size=12)
        y -= 15
    c.showPage()

    # === Key Insight Pages ===
    fields_to_plot = [
        "Final Score", "Company Age", "Number of employees 2023",
        "City Latin Alphabet", "Customer Segment", "Growth Category",
        "BvD sectors", "All Topics"
    ]

    for field in fields_to_plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        series = comparison_df[field]
        title = f"Distribution: {field}"
        color_match = "orange"
        color_input = "lightblue"

        if field in ["Final Score", "Company Age", "Number of employees 2023"]:
            values = series.tolist()
            labels = comparison_df["Company name Latin alphabet"]
            colors = [color_input if input_is_present and idx == "INPUT" else color_match for idx in comparison_df.index]
            
            ax.barh(labels, values, color=colors)
            ax.set_title(title)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            
            max_value = np.nanmax(values) if len(values) > 0 else 1
           
            if not np.isfinite(max_value) or max_value <= 0:
                max_value = 1
            
            threshold = 0.15 * max_value
            min_offset = 0.1
            
            for i, val in enumerate(values):
                if pd.notna(val):
                    label = f"{val:.2f}" if field == "Final Score" else f"{val:.0f}"
                    if val < threshold:
                        label_x = val + max(0.05 * max_value, min_offset)
                        label_align = "left"
                    else:
                        label_x = val - max(0.05 * max_value, min_offset)
                        label_align = "right"
                    ax.text(label_x, i, label, va="center", ha=label_align, fontsize=8, color="black", clip_on=False)
            
            ax.set_xlim(0, max_value * 1.2)
            fig.subplots_adjust(left=0.35)

        elif field in ["City Latin Alphabet", "Customer Segment", "Growth Category"]:
            vc = series.fillna("N/A").value_counts()
            labels = vc.index.tolist()
            match_counts = vc.tolist()
            input_val = comparison_df.loc["INPUT", field] if input_is_present and "INPUT" in comparison_df.index else None
            input_counts = [1 if label == input_val else 0 for label in labels]

            ax.barh(labels, match_counts, label="Matches", color=color_match)
            if input_val in labels:
                ax.barh(labels, input_counts, label="Input", color=color_input)
            else:
                ax.barh(["(not in top)"], [0], label="Input", color=color_input)
            ax.legend()
            ax.set_title(title)
            ax.invert_yaxis()
        elif field in ["All Topics", "BvD sectors"]:
            label_counts = defaultdict(lambda: {"INPUT": 0, "MATCH": 0})
            for idx, val in series.items():
                if pd.isna(val): 
                    continue
                labels_list = [t.strip() for t in val.split(";") if t.strip()]
                for label in labels_list:
                    if input_is_present and idx == "INPUT":
                        label_counts[label]["INPUT"] += 1
                    else:
                        label_counts[label]["MATCH"] += 1
            top_terms = sorted(label_counts.items(), key=lambda x: x[1]["INPUT"] + x[1]["MATCH"], reverse=True)[:10]
            if top_terms:
                labels_list = [x[0] for x in top_terms]
                match_counts = [x[1]["MATCH"] for x in top_terms]
                input_counts = [x[1]["INPUT"] for x in top_terms]
                bar_width = 0.6
                ax.barh(labels_list, match_counts, color=color_match, label="Matches", height=bar_width)
                # Added label parameter for INPUT to ensure it shows in the legend.
                ax.barh(labels_list, input_counts, color=color_input, left=match_counts, height=bar_width, label="Input")
                ax.set_title(f"Top 10 {field}")
                ax.legend()
                ax.invert_yaxis()

        fig.tight_layout()
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png", dpi=150)
        plt.close(fig)
        img_buf.seek(0)
        img = ImageReader(img_buf)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, H - 50, f"Key Insight: {field}")
        c.setFont("Helvetica", 12)
        field_summary = field_insights.get(field, f"{field} has varied values across the dataset.")
        # Use the auto-wrap function to ensure the summary text is wrapped.
        _ = draw_wrapped_text(c, field_summary, 50, H - 70, W - 100, font_size=12)
        c.drawImage(img, 40, H / 2 - 100, width=W - 80, height=H / 2)
        c.showPage()

    # === Strategic Recommendations & Caveats ===
    y = H - 50
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Strategic Recommendations")
    y -= 20
    c.setFont("Helvetica", 12)
    # Print the third summary paragraph (without numbering)
    y = draw_wrapped_text(c, summary_lines[2].strip(), 50, y, max_width=W - 100)
    y -= 20
    # If a fourth paragraph exists (caveats), print them; otherwise, skip.
    if len(summary_lines) >= 4:
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, y, "Caveats")
        y -= 20
        c.setFont("Helvetica", 12)
        y = draw_wrapped_text(c, summary_lines[3].strip(), 50, y, max_width=W - 100)
    c.showPage()

    c.save()
    buf.seek(0)
    return buf

# === Helper: generate download button ===
def get_download_button(df: pd.DataFrame, filename: str = "comparison.csv"):
    """
    Convert a pandas DataFrame to CSV bytes and render a Streamlit download button,
    keeping the DataFrame's index in the CSV.
    """

    csv_bytes = df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="📥 Download Comparison Table",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv"
    )

# === Load data & embeddings ===
@st.cache_data
def load_data():
    df = pd.read_csv("Sweden_final_filtered.csv")
    pa_embeddings = np.load("Sweden_product_activity_embeddings.npy")
    return df, pa_embeddings

df, pa_embeddings = load_data()

# 💬 Chat history state (must come first to avoid sidebar KeyError)
if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = {}  # {chat_id: {"name": ..., "messages": [...]}}
if "active_chat_id" not in st.session_state:
    st.session_state["active_chat_id"] = None
if "chat_counter" not in st.session_state:
    st.session_state["chat_counter"] = 1

# === Sidebar===
with st.sidebar:
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        st.image("your_logo.png", width=200)
        st.header("Competitor Chatbot for Sweden Consistent HGFs")
    
    st.markdown("""
    ## 🔍 How It Works:
    
    Find your closest competitors in Sweden's high-growth companies:
    
    - **Quick Match**: Enter your company details
    - **Smart Results**: Get top 5 similar companies
    - **Fine-tune**: Use filters to refine matches
    
    ## 📝 How to Describe Your Company:
    
    For best results, include:
    
    - **Core Info**: Products, activities, employee count
    - **Extra Details**: Year, location, legal form
    - **Company Name**: Full official name preferred
    """)
    st.markdown("---")

    st.markdown("## 💬 Chat History")

    with st.expander("📁 All Sessions", expanded=True):
        for chat_id, session in st.session_state["chat_sessions"].items():
            if st.button(session["name"], key=f"chat_{chat_id}"):
                st.session_state["active_chat_id"] = chat_id
                st.session_state["messages"] = session["messages"]

        if st.button("➕ New Chat"):
            new_id = f"chat_{st.session_state['chat_counter']}"
            st.session_state["chat_sessions"][new_id] = {
                "name": f"Chat {st.session_state['chat_counter']}",
                "messages": []
            }
            st.session_state["chat_counter"] += 1
            st.session_state["active_chat_id"] = new_id
            st.session_state["messages"] = []

            st.session_state["top5_companies"] = None
            st.session_state["input_company"] = None
            st.session_state["report_pdf_buf"] = None
            st.session_state["user_embedding"] = None
            st.session_state["show_filter_ui"] = False
            st.session_state["show_compare_table"] = False

            st.rerun()
    
# === Global Style Overrides
st.markdown("""
<style>
.main .block-container {
    max-width: 95vw;
    padding-left: 2vw;
    padding-right: 2vw;
}
section[data-testid="stChatInput"] {
    max-width: 100% !important;
    padding-left: 1vw;
    padding-right: 1vw;
}
section[data-testid="stChatInput"] textarea {
    min-height: 140px !important;
    font-size: 1rem;
}
div[data-testid="stVerticalBlock"] > div {
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# === Session State Initialization ===

# 💬 Chat message log (for current session)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 🔄 UI + Logic control states
if "result_block_counter" not in st.session_state:
    st.session_state["result_block_counter"] = 0
if "show_filter_ui" not in st.session_state:
    st.session_state["show_filter_ui"] = False
if "show_compare_table" not in st.session_state:
    st.session_state["show_compare_table"] = False
if "current_result_id" not in st.session_state:
    st.session_state["current_result_id"] = -1
if "triggered_block_id" not in st.session_state:
    st.session_state["triggered_block_id"] = -2

# === Insert Welcome Message (only once)
if not any(msg.get("content") == "Welcome message" for msg in st.session_state["messages"]):
    st.session_state["messages"].insert(0, {"role": "assistant", "content": "Welcome message"})

# === Display Chat History (render welcome separately)
for msg in st.session_state["messages"]:
    if msg["content"] == "Welcome message":
        with st.chat_message("assistant"):
            st.markdown("Hello! Please input your company name and description, and I'll match you with the most relevant competitors")
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# === Single Input Box (bottom) — only one box
prompt = st.chat_input("Please input your questions here.")

# === Utility Functions ===

def clean_description(text):
    return re.sub(r"\s+", " ", text.strip())

def normalize_company_name(name: str) -> str:

    return re.sub(r'\s+', ' ', name.strip()).lower()


def extract_company_name(desc):
    return desc.split(",")[0].strip() if "," in desc else desc.split()[0].strip()

def is_same_company(input_name, candidate_name):
    return input_name.lower().strip() in candidate_name.lower().strip() or \
           candidate_name.lower().strip() in input_name.lower().strip()

def extract_group_prefix(name):
    return name.split()[0].lower().strip()

# === Add new session to chat history if needed

# ✅ Move function definition above use
def generate_chat_title(text, max_words=6):
    if not text or not isinstance(text, str):
        return f"Chat {st.session_state.get('chat_counter', 1)}"
    cleaned = re.sub(r"\s+", " ", text.strip())
    words = cleaned.split()
    return " ".join(words[:max_words]) if words else f"Chat {st.session_state.get('chat_counter', 1)}"

# ✅ Only create new chat when needed
if st.session_state["active_chat_id"] is None:
    new_id = f"chat_{st.session_state['chat_counter']}"
    title = generate_chat_title(prompt)
    st.session_state["chat_sessions"][new_id] = {
        "name": title,
        "messages": []
    }
    st.session_state["chat_counter"] += 1
    st.session_state["active_chat_id"] = new_id

def extract_company_or_group_prefix(text):
    prompt = f"""
From the following business description, extract the **official full company name** as registered in Sweden.
Be sure to include suffixes like 'AB', '(Publ)', or any legal form if present.

Return only the exact name in plain text.

Description:
\"\"\"{text}\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract business group or company names."},
                {"role": "user", "content": prompt}
            ]
        )
        name = response.choices[0].message.content.strip()
        return name
    except:
        return ""

def extract_product_and_activity(text):
    prompt = f"""You are a business analyst. Given this company description, extract the business information in the format below:

Description: \"{text}\"

Return this:

Product offerings: ...
Key activities: ...

Only include concrete products and actions. Keep it concise."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert business analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        output = response.choices[0].message.content
        prod, act = "", ""
        if "Product offerings:" in output:
            parts = output.split("Product offerings:")[1].split("Key activities:")
            prod = parts[0].strip()
            if len(parts) > 1:
                act = parts[1].strip()
        return f"Products: {prod} | Activities: {act}"
    except:
        return ""

def get_openai_embedding(text):
    try:
        return client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding
    except:
        return np.zeros(1536)

def generate_natural_explanation(row):
    product = row.get("Product offerings", "")
    activity = row.get("Key activities", "")
    product_part = ""
    activity_part = ""

    if isinstance(product, str) and product.strip():
        product_part = f"The company offers {product.strip().lower()}"
    if isinstance(activity, str) and activity.strip():
        activity_part = f"and focuses on {activity.strip().lower()}"

    if product_part or activity_part:
        explanation = f"{product_part} {activity_part}".strip().capitalize()
        if not explanation.endswith("."):
            explanation += "."
        return explanation
    else:
        return "⚠️ This company lacks clear product and activity information."

def render_comparison_aggrid(df):
    # Transpose the DataFrame so each column is a company
    df_display = df.T.reset_index()
    df_display = df_display.rename(columns={"index": "Field"})

    gb = GridOptionsBuilder.from_dataframe(df_display)

    # Global default column behavior
    gb.configure_default_column(
        wrapText=True,
        autoHeight=True,
        resizable=True,
        cellStyle={
            "whiteSpace": "normal",
            "overflowWrap": "break-word"
        }
    )

    # Field column: narrow, bold, pinned
    gb.configure_column(
        "Field",
        pinned="left",
        width=120,
        minWidth=120,
        maxWidth=120,
        cellStyle={
            "fontWeight": "bold",
            "whiteSpace": "normal",
            "overflowWrap": "break-word"
        }
    )

    # Configure all other columns with fixed width
    for col in df_display.columns[1:]:
        gb.configure_column(
            col,
            width=160,
            minWidth=160,
            maxWidth=160,
            cellStyle={
                "whiteSpace": "normal",
                "overflowWrap": "break-word"
            }
        )

    gb.configure_grid_options(domLayout="normal")

    go = gb.build()

    st.markdown("### 🧾 Company Comparison Table")
    AgGrid(
        df_display,
        gridOptions=go,
        height=600,                     
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        theme="streamlit"
    )

# === Main chatbot logic ===

# === Run matching if prompt exists
if prompt and prompt.strip():
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    cleaned = clean_description(prompt)
    extracted_text = extract_product_and_activity(cleaned)
    user_embedding = get_openai_embedding(extracted_text)
    st.session_state["user_embedding"] = user_embedding

    company_or_group_name = extract_company_or_group_prefix(prompt)
    group_prefix = company_or_group_name.lower().split()[0] if company_or_group_name else ""
    st.session_state["input_company"] = company_or_group_name

    df_filtered = df.copy()
    df_filtered = df_filtered[~df_filtered["Company name Latin alphabet"].apply(lambda x: is_same_company(company_or_group_name, x))]
    df_filtered = df_filtered[~df_filtered["Company name Latin alphabet"].str.lower().str.startswith(group_prefix)]

    similarities = cosine_similarity([user_embedding], pa_embeddings[df_filtered.index])[0]
    df_filtered["Product/Activity Similarity"] = similarities

    emp_values = df_filtered["Number of employees 2023"]
    emp_log = np.log1p(emp_values)
    emp_scaled = (emp_log - emp_log.min()) / (emp_log.max() - emp_log.min())

    # Compute final score using a weighted combination of semantic similarity and company size
    df_filtered["Final Score"] = 0.9 * df_filtered["Product/Activity Similarity"] + 0.1 * emp_scaled

    # Increase threshold to 0.5 so that only candidates with a higher score are retained
    df_filtered = df_filtered[df_filtered["Final Score"] >= 0.5]

    # --- Add keyword-based filtering for further precision ---
    def simple_keyword_filter(input_text, candidate_text, threshold=0.3):
        """
        Extracts all words (in lowercase) from the input and candidate texts.
        Computes the proportion of input words found in the candidate text.
        Returns True if the ratio is greater than or equal to the threshold.
        """
        input_keywords = set(re.findall(r'\w+', input_text.lower()))
        candidate_keywords = set(re.findall(r'\w+', candidate_text.lower()))
        if not input_keywords:
            return False
        intersection_ratio = len(input_keywords.intersection(candidate_keywords)) / len(input_keywords)
        return intersection_ratio >= threshold

    filtered_candidates = []
    for idx, row in df_filtered.iterrows():
        # Use "Final Company Description" as the candidate text for keyword matching.
        candidate_text = row.get("Final Company Description", "")
        if simple_keyword_filter(prompt, candidate_text):
            filtered_candidates.append(row)

    df_filtered = pd.DataFrame(filtered_candidates)
    # --- End keyword-based filtering ---

    bot_reply = ""
    user_idx = df[df["Company name Latin alphabet"].str.lower().str.contains(company_or_group_name.lower())].index
    if len(user_idx) > 0:
        if np.all(pa_embeddings[user_idx[0]] == 0):
            bot_reply += "⚠️ The input company lacks structured product and activity information in the database. Matching results may be less reliable.\n\n"

    if df_filtered.empty:
        bot_reply += "🤔 No strong competitors found. Try describing the company's product and activity more clearly."
    else:
        top5 = df_filtered.sort_values(by="Final Score", ascending=False).head(5)
        st.session_state["top5_companies"] = top5
        st.session_state["input_company"] = company_or_group_name

        # Optional warning if the input company is not found in the database
        input_name = st.session_state["input_company"]
        input_normalized = normalize_company_name(input_name)

        matched_row = df[df["Company name Latin alphabet"].astype(str)
                         .apply(lambda x: normalize_company_name(x) == input_normalized)]
        if matched_row.empty:
            bot_reply += "⚠️ The input company was not found in the database. INPUT column will not appear in the comparison table.\n\n"
        for _, row in top5.iterrows():
            bot_reply += f"""
**🏢 {row['Company name Latin alphabet']}**  
**🔥 Final Score:** {round(row['Final Score'], 2)}  
**💡 Why is this a competitor?**  
{generate_natural_explanation(row)}  
_This match is based on semantic similarity in products and key activities, with additional weighting based on company size (number of employees)._  
📄 **Full Company Description:**  
{row['Final Company Description']}\n&nbsp;
"""
        bot_reply += """
What would you like to do next?

- 📊 **Compare Companies** — view a side-by-side comparison of the top matches  
- 🔍 **Filter Further** — narrow down the results by company size, location, legal form, etc.  
- 📥 **Download Table (CSV)** — export the full comparison table for further use
- 📄 **Generate Analysis Report** — get a visual analysis and full summary and can choose to download PDF format  
"""
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    if st.session_state["active_chat_id"]:
        title = generate_chat_title(prompt)
        st.session_state["chat_sessions"][st.session_state["active_chat_id"]]["name"] = title
    
# === Compare Companies Button ===
top5 = st.session_state.get("top5_companies", None)
user_input_exists = any(m["role"] == "user" for m in st.session_state.get("messages", []))

if top5 is not None and not top5.empty:
    compare_button = st.button("📊 Compare Companies")

    if compare_button:
        compare_fields = [
            "Company name Latin alphabet", "Region in country clean", "City Latin Alphabet",
            "Founded Year", "Standardized legal form", "BvD sectors", "Number of employees 2023",
            "Company Age", "Product offerings", "Key activities", "Revenue Streams",
            "Competitive Position", "Customer Segment", "All Topics", "Growth Category"
        ]

        raw_input_name = st.session_state.get("input_company", "")
        input_name = raw_input_name.strip().lower() if isinstance(raw_input_name, str) else ""

        input_row = df[df["Company name Latin alphabet"].str.strip().str.lower() == input_name]

        if input_row.empty:
            # No input company found – compare only MATCH 1–5
            comparison_df = top5[compare_fields].copy()
            comparison_df.index = [f"MATCH {i+1}" for i in range(len(comparison_df))]
        else:
            # Add INPUT + MATCH 1–5
            input_row = input_row[compare_fields]
            comparison_df = pd.concat(
                [input_row, top5[compare_fields]],
                ignore_index=True
            )
            comparison_df.index = ["INPUT"] + [f"MATCH {i+1}" for i in range(len(top5))]

        render_comparison_aggrid(comparison_df)

elif user_input_exists:
    st.warning("⚠️ No Top-5 companies found. Please input a description to start.")

# === Filter Further Button ===
if st.session_state.get("top5_companies") is not None and not st.session_state["top5_companies"].empty:
    if "show_filter_ui" not in st.session_state:
        st.session_state["show_filter_ui"] = False
    if "show_compare_table" not in st.session_state:
        st.session_state["show_compare_table"] = False

    if st.button("🔎 Filter Further"):
        st.session_state["show_filter_ui"] = True
        st.session_state["show_compare_table"] = False

    if st.session_state["show_filter_ui"]:
        st.markdown("### 🎛️ Refine Competitor Filters")
        selected_options = {}

        # Row 1: Numeric sliders
        col1, col_spacer, col2 = st.columns([5, 1, 5])
        with col1:
            emp_col = "Number of employees 2023"
            emp_range = st.slider(
                "👤 Number of Employees",
                int(df[emp_col].min()),
                int(df[emp_col].max()),
                (int(df[emp_col].min()), int(df[emp_col].max())),
                step=1
            )
            selected_options[emp_col] = emp_range

        with col2:
            year_col = "Founded Year"
            year_range = st.slider(
                "📅 Founded Year",
                int(df[year_col].min()),
                int(df[year_col].max()),
                (int(df[year_col].min()), int(df[year_col].max())),
                step=1
            )
            selected_options[year_col] = year_range

        # Row 2+: Multiselect filters
        filter_fields = {
            "City Latin Alphabet": "📍 City",
            "Standardized legal form": "🏛 Legal Form",
            "BvD sectors": "🏭 Sector",
            "Customer Segment": "👥 Customer Segment",
            "Growth Category": "🚀 Growth Category",
            "Region in country clean": "🌍 Region"
        }

        def multiselect_with_all(label, options, key):
            display_options = ["All"] + options
            selected = st.multiselect(label, display_options, key=key)
            return None if "All" in selected or not selected else selected

        filter_items = list(filter_fields.items())
        for i in range(0, len(filter_items), 3):
            row = st.columns(3)
            for j, (col, label) in enumerate(filter_items[i:i+3]):
                with row[j]:
                    options = sorted(df[col].dropna().unique())
                    selected = multiselect_with_all(label, options, key=f"filter_{col}")
                    if selected is not None:
                        selected_options[col] = selected

        # === Apply Filter
        if st.button("🚀 Apply Filter and Rerun Matching"):
            st.session_state["show_filter_ui"] = False
            user_embedding = st.session_state["user_embedding"]
            input_name = st.session_state.get("input_company", "")

            df_filtered = df.copy()
            for col, val in selected_options.items():
                if isinstance(val, tuple):  # numeric
                    df_filtered = df_filtered[(df_filtered[col] >= val[0]) & (df_filtered[col] <= val[1])]
                elif isinstance(val, list):
                    df_filtered = df_filtered[df_filtered[col].isin(val)]

            # Remove same company/group if applicable
            if input_name:
                df_filtered = df_filtered[
                    ~df_filtered["Company name Latin alphabet"].apply(lambda x: is_same_company(input_name, x))
                ]
                df_filtered = df_filtered[
                    ~df_filtered["Company name Latin alphabet"].str.lower().str.startswith(input_name.split()[0].lower())
                ]

            similarities = cosine_similarity([user_embedding], pa_embeddings[df_filtered.index])[0]
            df_filtered["Product/Activity Similarity"] = similarities
            df_filtered = df_filtered[df_filtered["Product/Activity Similarity"] >= 0.35]
            df_filtered["Final Score"] = df_filtered["Product/Activity Similarity"]

            top5_filtered = df_filtered.sort_values("Final Score", ascending=False).head(5)
            st.session_state["top5_companies"] = top5_filtered
            
            # === Output new response
            response_lines = []

            # Optional warning if input company is not found in database
            matched_row = df[df["Company name Latin alphabet"].str.lower().str.contains(input_name.lower())]
            if matched_row.empty:
                response_lines.insert(0, "⚠️ The input company was not found in the database. INPUT column will not appear in the comparison table.\n")

            if selected_options:
                tags = []
                for k, v in selected_options.items():
                    if isinstance(v, tuple):
                        tags.append(f"{k}: {v[0]}–{v[1]}")
                    elif isinstance(v, list):
                        tags.append(f"{k}: {', '.join(v)}")
                response_lines.append("🔎 **Filters applied:** " + "; ".join(tags))

            response_lines.append("Here are your **Top 5 Competitors after Filtering**:\n")
            for _, row in top5_filtered.iterrows():
                response_lines.append(f"""**🏢 {row['Company name Latin alphabet']}**  
**🔥 Final Score:** {round(row["Final Score"], 2)}  
**💡 Why is this a competitor?**  
{generate_natural_explanation(row)}  
📄 **Full Company Description:**  
{row['Final Company Description']}\n&nbsp;""")

            response_lines.append("Would you like to **filter further**, **compare these companies**, or **export this list**?")
            full_response = "\n\n".join(response_lines)

            with st.chat_message("assistant"):
                st.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Vertical layout (match your original behavior)
            if st.button("📊 Compare Companies", key=f"compare_btn_{len(st.session_state.messages)}"):
                st.session_state["show_compare_table"] = True
                st.session_state["show_filter_ui"] = False
            
            if st.button("🔎 Filter Further (Refine Again)", key=f"filter_btn_{len(st.session_state.messages)}"):
                st.session_state["show_filter_ui"] = True
                st.session_state["show_compare_table"] = False

# === Handle Compare Table (Works even if input company is missing) ===
if st.session_state.get("show_compare_table", False):
    input_name = st.session_state.get("input_company", "")
    top5 = st.session_state.get("top5_companies", None)

    if top5 is not None:
        matched_row = df[df["Company name Latin alphabet"].str.strip().str.lower() == input_name.strip().lower()]
        input_df = matched_row.iloc[[0]] if not matched_row.empty else pd.DataFrame(columns=df.columns)

        compare_fields = [
            "Company name Latin alphabet", "Region in country clean", "City Latin Alphabet",
            "Founded Year", "Standardized legal form", "BvD sectors", "Number of employees 2023",
            "Company Age", "Product offerings", "Key activities", "Revenue Streams",
            "Competitive Position", "Customer Segment", "All Topics", "Growth Category"
        ]

        if input_df.empty:
            # Placeholder row for missing INPUT
            input_df = pd.DataFrame([["–"] * len(compare_fields)], columns=compare_fields)

        comparison_df = pd.concat(
            [input_df[compare_fields], top5[compare_fields]],
            ignore_index=True
        )
        comparison_df.index = ["INPUT"] + [f"MATCH {i+1}" for i in range(len(top5))]
        render_comparison_aggrid(comparison_df)

    else:
        st.warning("⚠️ No Top-5 results to compare.")

    st.session_state["show_compare_table"] = False

# === Stable Download Button (Step 3) ===
if "top5_companies" in st.session_state and st.session_state["top5_companies"] is not None:
    top5 = st.session_state["top5_companies"]
    if top5 is None or top5.empty:
        st.warning("⚠️ No Top-5 companies found. Please input a description to start.")
        st.stop()  # Stop further execution
    
    raw_input_name = st.session_state.get("input_company", "")
    input_name = raw_input_name.strip().lower() if isinstance(raw_input_name, str) else ""

    matched_row = df[df["Company name Latin alphabet"]
                     .str.strip().str.lower() == input_name]
    
    if not matched_row.empty:
        input_df = matched_row
        comparison_df = pd.concat([input_df, top5], ignore_index=True)
        comparison_df.index = ["INPUT"] + [f"MATCH {i+1}" for i in range(len(top5))]
    else:
        comparison_df = top5.copy()
        comparison_df.index = [f"MATCH {i+1}" for i in range(len(top5))]

    # Render download button
    get_download_button(comparison_df, filename="sweden_competitors.csv")

# === PDF Download Section ===
if "top5_companies" in st.session_state and st.session_state["top5_companies"] is not None:
    top5 = st.session_state["top5_companies"]
    if top5 is None or top5.empty:
        st.warning("⚠️ No Top-5 companies found. Please input a description to start.")
        st.stop()  # Stop further execution
    
    if "report_pdf_buf" not in st.session_state:
        st.session_state["report_pdf_buf"] = None

    # When the "Generate Competitor Analysis Report (PDF)" button is clicked:
    if st.button("📄 Generate Competitor Analysis Report"):
        # Step 1: Construct comparison_df
        raw_input_name = st.session_state.get("input_company", "")
        input_name = raw_input_name.strip().lower() if isinstance(raw_input_name, str) else ""
        input_normalized = normalize_company_name(input_name)
        matched = df[df["Company name Latin alphabet"].apply(lambda x: normalize_company_name(x) == input_normalized)]
        
        if not matched.empty:
            comp_df = pd.concat([matched, top5], ignore_index=True)
            comp_df.index = ["INPUT"] + [f"MATCH {i+1}" for i in range(len(top5))]
        else:
            comp_df = top5.copy()
            comp_df.index = [f"MATCH {i+1}" for i in range(len(top5))]
        
        # Define input_is_present for use in the preview charts.
        input_is_present = comp_df.index[0] == "INPUT"
        
        # Step 2: Generate the PDF and store it in session_state.
        pdf_buf = generate_report_pdf(comp_df, st.session_state.get("input_company", "User Description"))
        st.session_state["report_pdf_buf"] = pdf_buf

        # Step 3: Generate a preview version of the report content.
        # Get the GPT-generated summary content.
        summary_lines, field_insights = gpt_generate_summary(top5, st.session_state.get("input_company", "User Description"), filters=None)
        preview_sections = []

        # Executive Summary (combines input description and first two summary paragraphs)
        executive_summary = f"**Executive Summary**\n\n"
        executive_summary += f"**Input Company Description:**\n{st.session_state.get('input_company', 'User Description')}\n\n"
        if len(summary_lines) >= 2:
            executive_summary += f"{summary_lines[0]}\n\n{summary_lines[1]}"
        preview_sections.append(executive_summary)

        # Generate Key Insights charts (one for each field)
        fields_to_plot = [
            "Final Score", "Company Age", "Number of employees 2023",
            "City Latin Alphabet", "Customer Segment", "Growth Category",
            "BvD sectors", "All Topics"
        ]
        insight_images = []
        for field in fields_to_plot:
            fig, ax = plt.subplots(figsize=(9, 5))
            series = comp_df[field]
            title = f"Distribution: {field}"
            color_match = "orange"
            color_input = "lightblue"
            
            if field in ["Final Score", "Company Age", "Number of employees 2023"]:
                values = series.tolist()
                labels = comp_df["Company name Latin alphabet"]
                colors = [color_input if input_is_present and idx == "INPUT" else color_match 
                          for idx in comp_df.index]
            
                ax.barh(labels, values, color=colors)
                ax.set_title(title)
                ax.set_yticklabels(labels, fontsize=9)
                ax.invert_yaxis()
            
                max_value = np.nanmax(values) if len(values) > 0 else 1

                if not np.isfinite(max_value) or max_value <= 0:
                    max_value = 1
            
                threshold = 0.15 * max_value
                min_offset = 0.2
            
                for i, val in enumerate(values):
                    if pd.notna(val):
                      
                        label = f"{val:.2f}" if field == "Final Score" else f"{val:.0f}"
                        if val < threshold:
                            label_x = val + max(0.05 * max_value, min_offset)
                            label_align = "left"
                        else:
                            label_x = val - max(0.05 * max_value, min_offset)
                            label_align = "right"
                        ax.text(label_x, i, label, va="center", ha=label_align, fontsize=8, 
                                color="black", clip_on=False)

                ax.set_xlim(0, max_value * 1.2)
            
                fig.subplots_adjust(left=0.35)

            elif field in ["City Latin Alphabet", "Customer Segment", "Growth Category"]:
                vc = series.fillna("N/A").value_counts()
                labels = vc.index.tolist()
                match_counts = vc.tolist()
                input_val = comp_df.loc["INPUT", field] if "INPUT" in comp_df.index else None
                input_counts = [1 if label == input_val else 0 for label in labels]
                ax.barh(labels, match_counts, label="Matches", color=color_match)
                if input_val in labels:
                    ax.barh(labels, input_counts, label="Input", color=color_input)
                else:
                    ax.barh(["(not in top)"], [0], label="Input", color=color_input)
                ax.legend()
                ax.set_title(title)
                ax.invert_yaxis()
                
            elif field in ["All Topics", "BvD sectors"]:
                label_counts = defaultdict(lambda: {"INPUT": 0, "MATCH": 0})
                for idx, val in series.items():
                    if pd.isna(val):
                        continue
                    labels_list = [t.strip() for t in val.split(";") if t.strip()]
                    for label in labels_list:
                        if input_is_present and idx == "INPUT":
                            label_counts[label]["INPUT"] += 1
                        else:
                            label_counts[label]["MATCH"] += 1
                top_terms = sorted(label_counts.items(), key=lambda x: x[1]["INPUT"] + x[1]["MATCH"], reverse=True)[:10]
                if top_terms:
                    labels_list = [x[0] for x in top_terms]
                    match_counts = [x[1]["MATCH"] for x in top_terms]
                    input_counts = [x[1]["INPUT"] for x in top_terms]
                    bar_width = 0.6
                    ax.barh(labels_list, match_counts, color=color_match, label="Matches", height=bar_width)
                    ax.barh(labels_list, input_counts, color=color_input, left=match_counts, height=bar_width, label="Input")
                    ax.legend()
                    ax.set_title(title)
                    ax.invert_yaxis()
            fig.tight_layout()
            buf_img = BytesIO()
            plt.savefig(buf_img, format="png", dpi=150)
            plt.close(fig)
            buf_img.seek(0)
            insight_images.append(buf_img)
        # Strategic Recommendations text (3rd summary paragraph)
        strategic_recs = ""
        if len(summary_lines) >= 3:
            strategic_recs = summary_lines[2]
        strategic_section = f"**Strategic Recommendations**\n\n{strategic_recs}"
        preview_sections.append(strategic_section)
        
        # Display the preview in the chat message.
        with st.chat_message("assistant"):
            st.markdown("\n\n".join(preview_sections))
            for img in insight_images:
                st.image(img)
            st.markdown("Below is your complete report in PDF format. Click the download button to save the file.")
            st.download_button(
                label="📄 Download Competitor Analysis Report (PDF)",
                data=st.session_state["report_pdf_buf"].getvalue(),
                file_name="competitive_analysis_report.pdf",
                mime="application/pdf"
            )
        st.session_state.messages.append({"role": "assistant", "content": "Report preview generated."})

    # If the PDF has already been generated, simply display the download button.
    elif st.session_state["report_pdf_buf"] is not None:
        st.download_button(
            label="📄 Download Competitor Analysis Report (PDF)",
            data=st.session_state["report_pdf_buf"].getvalue(),
            file_name="competitive_analysis_report.pdf",
            mime="application/pdf"
        )
