from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "models" / "stress_model.joblib"
METADATA_PATH = PROJECT_FOLDER / "models" / "model_metadata.json"

# Keep the original descriptions and add everyday explanations.
QUESTIONS = [
    (
        "TIPI1",
        "Extraverted, enthusiastic",
        "I am outgoing around people and show excitement about things.",
    ),
    (
        "TIPI2",
        "Critical, quarrelsome",
        "I tend to point out faults and get into arguments.",
    ),
    (
        "TIPI3",
        "Dependable, self-disciplined",
        "People can rely on me, and I can stay focused on my responsibilities.",
    ),
    (
        "TIPI4",
        "Anxious, easily upset",
        "I tend to worry and become upset easily.",
    ),
    (
        "TIPI5",
        "Open to new experiences, complex",
        "I enjoy new experiences and exploring different or detailed ideas.",
    ),
    (
        "TIPI6",
        "Reserved, quiet",
        "I tend to keep my thoughts to myself and speak less around others.",
    ),
    (
        "TIPI7",
        "Sympathetic, warm",
        "I care about other people's feelings and am friendly towards them.",
    ),
    (
        "TIPI8",
        "Disorganized, careless",
        "I tend to be unorganised and may not pay enough attention to details.",
    ),
    (
        "TIPI9",
        "Calm, emotionally stable",
        "I usually stay calm, and my emotions do not change easily.",
    ),
    (
        "TIPI10",
        "Conventional, uncreative",
        "I tend to follow familiar ways and do not often come up with new ideas.",
    ),
]

RESPONSE_OPTIONS = {
    1: "Strongly disagree",
    2: "Mostly disagree",
    3: "Slightly disagree",
    4: "Neither agree nor disagree",
    5: "Slightly agree",
    6: "Mostly agree",
    7: "Strongly agree",
}

EDUCATION_OPTIONS = {
    1: "Below high-school completion",
    2: "High school completed; no university degree yet",
    3: "Bachelor's degree completed — for example, BA, BSc or BTech",
    4: "Postgraduate degree completed — for example, MA, MSc, MTech or PhD",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)




def show_welcome_illustration():
    st.html("""
<style>
.stApp {background:#1d1e1c;color:#eeeae1}
.block-container {max-width:1140px;padding-top:5rem}
.editorial-nav {display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #55564e;padding:0 0 22px;margin-bottom:50px;color:#eeeae1;font-family:monospace;font-size:12px;letter-spacing:2px}
.editorial-brand {font-size:22px;letter-spacing:4px;line-height:1.2}
.editorial-hero {display:grid;grid-template-columns:1.25fr 1fr;gap:45px;align-items:center;margin-bottom:46px;color:#eeeae1}
.editorial-kicker {font:12px monospace;letter-spacing:2px;color:#b7c0a4;text-transform:uppercase}
.editorial-title {font-family:Arial,sans-serif;font-size:clamp(42px,6vw,78px);line-height:1.02;letter-spacing:-3px;font-weight:400;margin:28px 0}
.editorial-title em {font-family:Georgia,serif;font-weight:400;color:#c2c7ae}
.editorial-copy {font:17px/1.7 Arial,sans-serif;color:#c9c7be;max-width:440px}
.editorial-photo {width:100%;height:420px;object-fit:cover;filter:saturate(.55);border-radius:170px 170px 2px 2px}
.editorial-credit {font:11px/1.6 monospace;color:#b9baaf;margin-top:12px}
.editorial-credit a {color:inherit}
.editorial-rule {border-top:1px solid #55564e;margin:30px 0}
.editorial-flow {display:grid;grid-template-columns:1fr 1fr 1fr;gap:30px;padding:24px 0 40px;border-bottom:1px solid #55564e;margin-bottom:30px;color:#eeeae1}
.editorial-flow svg {height:66px;width:85px;stroke:#bfc5a6;stroke-width:1.7;fill:none;stroke-linecap:round;stroke-linejoin:round;margin-bottom:16px}
.editorial-flow h3 {font:18px Arial,sans-serif;margin:10px 0}
.editorial-flow p {font:14px/1.7 Arial,sans-serif;color:#bdbeb4;max-width:260px}
.editorial-index {font:11px monospace;letter-spacing:2px;color:#acb398}
@media(max-width:640px){.editorial-hero{grid-template-columns:1fr;gap:20px}.editorial-photo{height:260px;border-radius:110px 110px 0 0}.editorial-nav{margin-bottom:28px}.editorial-flow{grid-template-columns:1fr;gap:18px}.editorial-title{letter-spacing:-1.5px}.editorial-flow>div{border-bottom:1px solid #444;padding-bottom:12px}}
</style>
<div class="editorial-nav"><div class="editorial-brand">PAUSE<br>— &amp; REFLECT</div><div>STRESS SCORE<br>STUDENT PROJECT</div></div>
<div class="editorial-hero">
<div><div class="editorial-kicker">A simple student project</div>
<div class="editorial-title">Answer a few<br>questions.<br><em>See your estimate.</em></div>
<p class="editorial-copy">Tell us a little about yourself. This app uses a trained computer model to estimate a stress score. It is not a medical test.</p>
<svg width="150" height="60" viewBox="0 0 150 60" aria-hidden="true"><path d="M8 22C42 0 109 8 99 31C86 58 30 44 46 22C61 3 119 15 140 42M125 38L140 42L138 27" stroke="#bfc5a6" stroke-width="1.7" fill="none" stroke-linecap="round"/></svg>
<div class="editorial-kicker">10 questions · simple explanations</div></div>
<div><img class="editorial-photo" src="https://images.unsplash.com/photo-1688024611618-40ee8b0c45f8?auto=format&amp;fit=crop&amp;w=1000&amp;q=85" alt="A real photograph of fern leaves emerging from a dark background"/>
<div class="editorial-credit">A QUIET MOMENT<br>Photograph: <a href="https://unsplash.com/photos/lduDz1Ra0sY" target="_blank" rel="noopener noreferrer">Simon Reza / Unsplash</a></div></div>
</div>
""")

def show_process_diagram():
    st.html("""
<div class="editorial-rule"></div><div class="editorial-kicker">The process / three simple steps</div>
<div class="editorial-flow">
<div><svg viewBox="0 0 85 66" aria-hidden="true"><path d="M20 12L55 9L59 55L18 57Z M27 22L45 20 M27 30L42 28 M27 39L37 38 M47 43L67 17L73 22L53 48L45 51Z"/></svg><div class="editorial-index">01 / ANSWER</div><h3>Answer 10 questions.</h3><p>Choose how much each description fits you. There are no right or wrong answers.</p></div>
<div><svg viewBox="0 0 85 66" aria-hidden="true"><path d="M42 16V29M42 29L19 40M42 29L66 40M19 47V57M66 47V57"/><circle cx="42" cy="10" r="6"/><rect x="12" y="40" width="14" height="9"/><rect x="59" y="40" width="14" height="9"/><path d="M8 59H30M55 59H77"/></svg><div class="editorial-index">02 / ESTIMATE</div><h3>Get an estimated score.</h3><p>The computer uses what it learned from earlier survey answers.</p></div>
<div><svg viewBox="0 0 85 66" aria-hidden="true"><path d="M12 51H74M19 46V31M35 46V20M51 46V12M67 46V26M10 15Q21 3 33 9"/><circle cx="60" cy="16" r="14"/></svg><div class="editorial-index">03 / UNDERSTAND</div><h3>Read what it means.</h3><p>See the score on a 0–42 scale, alongside its meaning and the model's limitations.</p></div>
</div>
""")

def show_score_illustration(score):
    position = 50 + min(42, max(0, score)) / 42 * 700
    st.image(f"""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 140">
<rect width="800" height="140" rx="2" fill="#292c27"/>
<path d="M50 72H750" stroke="#65705e" stroke-width="2"/>
<path d="M50 65V79M400 65V79M750 65V79" stroke="#a7ae98" stroke-width="1"/>
<circle cx="{position}" cy="72" r="9" fill="#c5cfad"/>
<text x="{position}" y="43" text-anchor="middle" font-family="monospace" font-size="20" fill="#eeeae1">{score:.1f}</text>
<text x="50" y="111" font-family="monospace" font-size="14" fill="#d3d4c8">0</text>
<text x="400" y="111" text-anchor="middle" font-family="monospace" font-size="12" fill="#d3d4c8">ESTIMATED STRESS SCORE</text>
<text x="750" y="111" text-anchor="end" font-family="monospace" font-size="14" fill="#d3d4c8">42</text>
</svg>
""", alt=f"Estimated score {score:.1f} on a scale from 0 to 42.")
    st.caption("This position is an estimate, not a medical risk category.")


def save_answer(name):
    st.session_state["answers"][name] = st.session_state["_input_" + name]
    st.session_state.pop("result", None)


def go_button(label, page, key, primary=False):
    if st.button(label, key=key, type="primary" if primary else "secondary"):
        st.switch_page(page)


def home_page():
    show_welcome_illustration()
    st.caption("This project estimates a stress questionnaire score. It does not measure your overall mental health.")
    left, right = st.columns(2)
    with left:
        go_button("Start the questions", QUESTION_PAGE, "start", True)
    with right:
        go_button("Learn how it works", ABOUT_PAGE, "home_about")
    show_process_diagram()


def question_page():
    st.title("A few questions about you")
    st.write("Choose how much you agree with each sentence. Think about how you are in general. There are no right or wrong answers.")
    st.caption("Your answers stay available while you move between these pages in this session. The app does not write them to a file or database.")
    saved = st.session_state["answers"]
    st.subheader("1. Your age and education")
    for name in ["age", "education"] + [q[0] for q in QUESTIONS]:
        widget = "_input_" + name
        if widget not in st.session_state:
            st.session_state[widget] = saved.get(name)

    st.number_input("How old are you?", min_value=13, max_value=100,
                   value=None, step=1, key="_input_age",
                   on_change=save_answer, args=("age",))
    st.selectbox("What is the highest education level you have completed?",
                 options=list(EDUCATION_OPTIONS), index=None,
                 format_func=lambda x: EDUCATION_OPTIONS[x], key="_input_education",
                 on_change=save_answer, args=("education",))
    st.caption("Choose a level you have finished. If you are studying for a bachelor's degree, do not select it until you finish. Postgraduate means a qualification after a bachelor's degree.")
    st.subheader("2. What describes you?")
    st.write("Read each sentence, then choose your answer.")
    count = sum(saved.get(q[0]) is not None for q in QUESTIONS)
    st.caption(f"{count} of 10 questions answered")
    for number, (name, original, explanation) in enumerate(QUESTIONS, 1):
        st.selectbox(f"{number}. {explanation}", options=list(RESPONSE_OPTIONS),
                     index=None, format_func=lambda x: RESPONSE_OPTIONS[x],
                     placeholder="Choose an answer", key="_input_" + name,
                     help="Original survey wording: " + original,
                     on_change=save_answer, args=(name,))
    st.caption("These simple explanations help you read the original survey descriptions. They have not been tested as a new questionnaire.")
    if st.button("Next: see my result", type="primary"):
        needed = ["age", "education"] + [q[0] for q in QUESTIONS]
        missing = [x for x in needed if saved.get(x) is None]
        if missing:
            st.warning("Please enter your age, select your education, and answer all 10 questions.")
        elif not MODEL_PATH.exists() or not METADATA_PATH.exists():
            st.error("The trained model is missing. Run train_model.py in your project, then try again.")
        else:
            with METADATA_PATH.open(encoding="utf-8") as file:
                metadata = json.load(file)
            values = {x: saved[x] for x in needed}
            data = pd.DataFrame([values], columns=metadata["features"])
            score = float(load_model().predict(data)[0])
            st.session_state["result"] = {"score": score, "metadata": metadata}
            st.switch_page(RESULT_PAGE)
    go_button("Back to home", HOME_PAGE, "question_back")


def result_page():
    st.title("Your estimated stress score")
    result = st.session_state.get("result")
    if result is None:
        st.write("Answer the questions first to get your estimate. If you changed an answer, calculate your score again.")
        go_button("Go to the questions", QUESTION_PAGE, "result_missing", True)
        return
    score = result["score"]
    mae = result["metadata"]["test_metrics"]["MAE"]
    st.metric("Estimated score", f"{score:.1f} out of 42")
    show_score_illustration(score)
    st.subheader("What does this mean?")
    st.write("The survey score runs from 0 to 42. A higher score means more stress symptoms were reported on that survey. This number is not a percentage.")
    st.write("Your number is the computer's estimate. It is not a score measured by a health professional and does not tell you whether you have a mental health condition.")
    st.subheader("Can the estimate be wrong?")
    st.write(f"Yes. In testing, predictions were about {mae:.1f} points away from people's survey scores on average. Your estimate could be closer or further away. This is not a guaranteed range.")
    st.caption("If you are worried about how you feel, speak with a qualified mental health professional. Do not use this score to decide whether to seek help.")
    left, right = st.columns(2)
    with left:
        go_button("Back: change my answers", QUESTION_PAGE, "edit")
    with right:
        go_button("Next: learn about this project", ABOUT_PAGE, "result_about", True)


def about_page():
    st.title("About this project")
    st.subheader("What does the app do?")
    st.write("It estimates a stress score using your age, education and 10 answers about your personality. It is a student project showing how machine learning can make numerical predictions.")
    show_process_diagram()
    st.subheader("Where did the model learn from?")
    st.write("We used 39,775 responses from a public online survey. People answered personality questions and a stress questionnaire. We cleaned the data, trained models and checked predictions against separate survey responses.")
    st.subheader("What is the stress score?")
    st.write("The original questionnaire is called the Depression Anxiety Stress Scales, or DASS. This project uses only the stress part: 14 questions whose scores add up to a number between 0 and 42.")
    st.subheader("Why not just add up my answers?")
    st.write("The 10 personality questions you answer here are different from the 14 stress questions. The computer estimates the stress score from those separate inputs. It does not simply add your answers together.")
    st.subheader("What should I keep in mind?")
    st.write("The survey was voluntary and may not represent everyone. The model has not been checked specifically for Indian students. It cannot diagnose a condition or predict your future mental health.")
    with st.expander("For my teacher: model and test results"):
        st.write("We compared a model that always predicts the average, Ridge regression and Random Forest regression. Random Forest performed best on the validation data, which we used to choose the model.")
        st.write("Random Forest combines estimates from many decision trees. A decision tree uses a sequence of questions about the inputs to make a prediction.")
        if METADATA_PATH.exists():
            with METADATA_PATH.open(encoding="utf-8") as file:
                metadata = json.load(file)
            metrics = metadata["test_metrics"]
            st.write(f"MAE: {metrics['MAE']:.3f} points — average size of prediction errors. Lower is better.")
            st.write(f"RMSE: {metrics['RMSE']:.3f} points — an error measure that gives more weight to bigger mistakes. Lower is better.")
            st.write(f"R²: {metrics['R2']:.3f} — how much variation in test scores the model explains compared with predicting the test-set average. This is not an accuracy percentage.")
            st.write(f"We used {metadata['development_rows']:,} rows for final training and {metadata['test_rows']:,} separate rows for final testing.")
        else:
            st.caption("Train the model to see saved test results here.")
        st.write("Missing inputs were filled using values learned only from training data. The questionnaire items used to calculate the target score were excluded from the inputs.")
        st.write("The test results are from the original survey, not a clinical evaluation of this app or its simpler explanations.")
    go_button("Next: see the model graphs", PERFORMANCE_PAGE, "about_graphs", True)
    st.link_button("Open the original dataset website", "https://openpsychometrics.org/_rawdata/")
    left, right = st.columns(2)
    with left:
        go_button("Back to home", HOME_PAGE, "about_home")
    with right:
        go_button("Go to the questions", QUESTION_PAGE, "about_questions", True)



@st.cache_data
def test_plot_data(data_stamp, model_stamp, metadata_stamp):
    # Recreate the original held-out split. Never fit or tune on this page.
    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)
    frame = pd.read_csv(PROJECT_FOLDER / "data" / "prepared_data.csv")
    if len(frame) != metadata["development_rows"] + metadata["test_rows"]:
        raise ValueError("The dataset size differs from the data used for training.")
    _, test = train_test_split(frame, test_size=0.20,
                               random_state=metadata["random_state"])
    # Load here rather than use the prediction cache, so a new file is respected.
    model = joblib.load(MODEL_PATH)
    predicted = model.predict(test[metadata["features"]])
    actual = test["stress_score"].to_numpy()
    measured = {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
        "R2": r2_score(actual, predicted),
    }
    for key, value in measured.items():
        if not np.isclose(value, metadata["test_metrics"][key], rtol=1e-6, atol=1e-6):
            raise ValueError("The test results do not match the saved training results. Use the original prepared dataset and matching model files.")
    return pd.DataFrame({"Actual": actual, "Predicted": predicted,
                         "Error": predicted - actual})


def plot_frame():
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor="#1d1e1c")
    ax.set_facecolor("#1d1e1c")
    ax.tick_params(colors="#dddcd2")
    ax.xaxis.label.set_color("#dddcd2")
    ax.yaxis.label.set_color("#dddcd2")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#6a7162")
    ax.grid(axis="y", alpha=.12, color="#d5dcc7")
    ax.set_axisbelow(True)
    return fig, ax


def display_plot(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def performance_page():
    st.title("How well does the model work?")
    st.write("These graphs show the saved model's test results. They describe the project, not your personal mental health.")
    charts = [
        ("Which model made smaller mistakes?", "comparison.png",
         "Shorter bars are better. These are average errors on the same validation data, used to choose the model."),
        ("How close were the final predictions?", "predictions.png",
         "Each dot is one test response. Dots closer to the dotted line mean better predictions. This is a fixed sample of up to 1,500 test responses."),
        ("Were predictions too high or too low?", "errors.png",
         "Zero means no error. Left of zero means the prediction was too low; right means it was too high. This uses all test responses."),
        ("Which scores were in the test data?", "scores.png",
         "Taller bars mean more responses with that score. This survey sample does not represent everyone.")
    ]
    for title, filename, explanation in charts:
        st.subheader(title)
        st.write(explanation)
        path = PROJECT_FOLDER / "assets" / filename
        if path.exists():
            st.image(str(path), alt=title)
        else:
            st.warning("This graph file is missing from the deployment.")
    st.caption("Charts were exported after checking the original held-out results against the model metadata. The hosted app does not need the raw dataset.")
    go_button("Back to About", ABOUT_PAGE, "graphs_back")
    go_button("Go to the questions", QUESTION_PAGE, "graphs_questions", True)


def main():
    global HOME_PAGE, QUESTION_PAGE, RESULT_PAGE, ABOUT_PAGE, PERFORMANCE_PAGE
    st.set_page_config(page_title="Pause & Reflect | Stress score",
                       page_icon=":material/adjust:", layout="centered")
    st.session_state.setdefault("answers", {})
    HOME_PAGE = st.Page(home_page, title="Home", icon=":material/home:", default=True)
    QUESTION_PAGE = st.Page(question_page, title="Questions", icon=":material/edit:", url_path="questions")
    RESULT_PAGE = st.Page(result_page, title="Result", icon=":material/bar_chart:", url_path="result")
    ABOUT_PAGE = st.Page(about_page, title="About", icon=":material/info:", url_path="about")
    PERFORMANCE_PAGE = st.Page(performance_page, title="Model performance", icon=":material/bar_chart:", url_path="performance")
    page = st.navigation([HOME_PAGE, QUESTION_PAGE, RESULT_PAGE, ABOUT_PAGE, PERFORMANCE_PAGE], position="top")
    page.run()


if __name__ == "__main__":
    main()
