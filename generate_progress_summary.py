from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
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
        (
            "Iterated Prisoner’s Dilemma (IPD)",
            "Functional evolutionary simulation with 30 generations and 64 agents. Generates strategy-distribution, \n"
            "Pareto efficiency, Nash deviation and action-regret plots. *Next step:* incorporate LLM-driven adaptive strategies."),
        ("MIT Beer Game",
         "Robust supply-chain simulator with inventory, backlog and profit tracking. Memory & communication scaffolding \n"
         "implemented; preliminary 100-round stability run completed. *Next step:* parameter sweep on memory length and \n"
         "shared vs individual memory."),
        ("Fishery Game",
         "Implements logistic stock growth, extraction decisions and inequality metrics. Resource collapses under aggressive \n"
         "harvest – highlighting need for incentive redesign. *Next step:* agent heterogeneity and side-payments."),
        ("Market Impact Game",
         "Baseline BUY/SELL/HOLD agents completed; LLM trader class stubbed. *Next step:* evaluate market depth feedback loops."),
        ("Oligopoly Simulation",
         "Price grid, demand noise and cost asymmetry variations ready; LLM agents optional. *Next step:* measure time-to-collusion \n"
         "with different temperature settings."),
        ("Chinese Whispers SQL (v1 / v2)",
         "End-to-end pipeline: story rewrite → SQL generation → execution drift measurement. Organised results, story library, \n"
         "LangGraph orchestration, LangSmith tracing in place. *Next step:* expand student DB and add automated unit tests."),
        ("Security Dilemma",
         "Core turn-based simulation finished. No communication or memory yet. *Next step:* enable multi-agent chat before \n"
         "each arms-investment decision."),
    ]

    for title, desc in game_summaries:
        flow.append(Paragraph(f"<b>{title}</b>: {desc}", styles["BodyText"]))
        flow.append(Spacer(1, 0.15 * inch))

    flow.append(PageBreak())

    # --- Page 3: Key Results & Future Work ---
    flow.append(Paragraph("Key Simulation Outputs", styles["Heading2Center"]))
    flow.append(Spacer(1, 0.15 * inch))

    # Helper to add image if exists
    def add_image(path, max_width=6 * inch, max_height=7 * inch):
        if not os.path.isfile(path):
            flow.append(Paragraph(f"<i>Image not found: {path}</i>", styles["BodyText"]))
            flow.append(Spacer(1, 0.1 * inch))
            return
        img = Image(path)
        # Scale proportionally
        width, height = img.imageWidth, img.imageHeight
        ratio = 1.0
        if width > max_width:
            ratio = max_width / width
        if height * ratio > max_height:
            ratio = max_height / height
        img.drawWidth = width * ratio
        img.drawHeight = height * ratio
        flow.append(img)
        flow.append(Spacer(1, 0.2 * inch))

    # Paths to plots
    plots = [
        "Games/1_Prisoners_Dilemma/simulation_results/run_2025-03-24_18-34-55-30-Runs-with-64-Agents/research_metrics.png",
        "Games/2_MIT_Beer_Game/simulation_results/ Stable run 100/combined_plots.png",
        "Games/2_MIT_Beer_Game/simulation_results/run_163_2025-06-30_18-52-23/combined_plots.png",
        "Games/3_Fishery_Game/fishery_simulation_results/run_2025-05-09_10-17-38/comprehensive_metrics.png",
    ]

    for p in plots:
        add_image(p)

    flow.append(Spacer(1, 0.2 * inch))
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