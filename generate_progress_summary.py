from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os


def build_pdf(output_path: str):
    """Generate a three-page progress summary PDF with graphs."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Heading1Center", parent=styles["Heading1"], alignment=1))
    styles.add(ParagraphStyle(name="Heading2Center", parent=styles["Heading2"], alignment=1))

    flow = []

    # --- Page 1: Title & Overview ---
    flow.append(Paragraph("Multi-Agent Equilibria – Progress Report", styles["Heading1Center"]))
    flow.append(Spacer(1, 0.2 * inch))

    overview_text = (
        "This report summarises the work completed to date on the *Multi-Agent Equilibria* research project. "
        "The objective of the project is to explore emergent strategic behaviour in large-language-model (LLM) agents "
        "across a suite of classic game-theoretic environments and supply-chain simulations. "
        "The current codebase implements eight distinct games, extensive logging & visualisation utilities, and early support "
        "for LangGraph-based workflow orchestration. While each simulation is functional, the emphasis so far has been on "
        "building *foundational* pipelines rather than exhaustive scientific evaluation."
    )
    flow.append(Paragraph(overview_text, styles["BodyText"]))
    flow.append(Spacer(1, 0.25 * inch))

    repo_scope = (
        "The repository now contains:\n"
        "• **Iterated Prisoner’s Dilemma** (evolutionary, 8 classic strategies)\n"
        "• **MIT Beer Game** (four-tier supply-chain, memory & communication hooks)\n"
        "• **Fishery Game** (common-pool resource extraction)\n"
        "• **Market Impact Game** (multi-agent trading)\n"
        "• **Oligopoly Simulation** (price-setting competition)\n"
        "• **Chinese Whispers SQL – v1 & v2** (story-to-SQL drift study)\n"
        "• **Security Dilemma** (arms-race interaction)"
    )
    flow.append(Paragraph(repo_scope, styles["BodyText"]))

    flow.append(PageBreak())

    # --- Page 2: Detailed game status ---
    flow.append(Paragraph("Game-by-Game Status", styles["Heading2Center"]))
    flow.append(Spacer(1, 0.2 * inch))

    game_summaries = [
        {
            "title": "Iterated Prisoner’s Dilemma (IPD)",
            "desc": "Functional evolutionary simulation with 30 generations and 64 agents. Generates strategy-distribution, Pareto efficiency, Nash deviation and action-regret plots. <i>Next step:</i> incorporate LLM-driven adaptive strategies.",
            "img": "Games/1_Prisoners_Dilemma/simulation_results/run_2025-03-24_18-34-55-30-Runs-with-64-Agents/research_metrics.png",
        },
        {
            "title": "MIT Beer Game – Stable Run 100",
            "desc": "Robust supply-chain simulator with inventory, backlog and profit tracking. Memory & communication scaffolding implemented; preliminary 100-round stability run completed. <i>Next step:</i> parameter sweep on memory length and shared vs individual memory.",
            "img": "Games/2_MIT_Beer_Game/simulation_results/ Stable run 100/combined_plots.png",
        },
        {
            "title": "MIT Beer Game – Run 163",
            "desc": "Same environment with revised parameters to stress-test communication constraints.",
            "img": "Games/2_MIT_Beer_Game/simulation_results/run_163_2025-06-30_18-52-23/combined_plots.png",
        },
        {
            "title": "Fishery Game",
            "desc": "Implements logistic stock growth, extraction decisions and inequality metrics. Resource collapses under aggressive harvest – highlighting need for incentive redesign. <i>Next step:</i> agent heterogeneity and side-payments.",
            "img": "Games/3_Fishery_Game/fishery_simulation_results/run_2025-05-09_10-17-38/comprehensive_metrics.png",
        },
        {
            "title": "Market Impact Game",
            "desc": "Baseline BUY/SELL/HOLD agents completed; LLM trader class stubbed. <i>Next step:</i> evaluate market-depth feedback loops.",
            "img": None,
        },
        {
            "title": "Oligopoly Simulation",
            "desc": "Price grid, demand noise and cost asymmetry variations ready; LLM agents optional. <i>Next step:</i> measure time-to-collusion with different temperature settings.",
            "img": None,
        },
        {
            "title": "Chinese Whispers SQL (v1 / v2)",
            "desc": "End-to-end pipeline: story rewrite → SQL generation → execution drift measurement. Organised results, story library, LangGraph orchestration, LangSmith tracing in place. <i>Next step:</i> expand student DB and add automated unit tests.",
            "img": None,
        },
        {
            "title": "Security Dilemma",
            "desc": "Core turn-based simulation finished. No communication or memory yet. <i>Next step:</i> enable multi-agent chat before each arms-investment decision.",
            "img": None,
        },
    ]

    table_data = []
    col_widths = [3.0 * inch, 3.5 * inch]

    def scaled_img(path: str):
        if path is None or not os.path.isfile(path):
            return ""
        img = Image(path)
        max_w, max_h = col_widths[1], 2.2 * inch
        ratio = min(max_w / img.imageWidth, max_h / img.imageHeight)
        img.drawWidth = img.imageWidth * ratio
        img.drawHeight = img.imageHeight * ratio
        return img

    for item in game_summaries:
        left_para = Paragraph(f"<b>{item['title']}</b>: {item['desc']}", styles["BodyText"])
        right_obj = scaled_img(item["img"])
        table_data.append([left_para, right_obj])

    summary_table = Table(table_data, colWidths=col_widths, hAlign="LEFT")
    summary_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    flow.append(summary_table)

    flow.append(Spacer(1, 0.25 * inch))

    # --- Future Work (brief) ---
    future_work = (
        "<b>Future Work:</b> We are exploring a shift from isolated game-theoretic environments toward a more holistic "
        "*world-model* in which heterogeneous LLM agents reason, plan and negotiate across interconnected sub-games. "
        "Initial experiments on a dedicated <i>LangGraph_Branch</i> are prototyping cross-game memory pools and event-driven "
        "workflow orchestration."
    )
    flow.append(Paragraph(future_work, styles["BodyText"]))

    # Build PDF
    doc.build(flow)


if __name__ == "__main__":
    output_pdf = "docs/Progress_Summary_July2025.pdf"
    build_pdf(output_pdf)
    print(f"PDF generated at {output_pdf}")