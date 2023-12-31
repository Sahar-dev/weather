import streamlit as st

st.set_page_config(
    page_title="Welcome to The Rain Prediction",
    page_icon=":partly_sunny:",
    layout="centered",
)

st.markdown(
    """
    <style>
        h1 {
            position: relative;
            padding: 0;
            margin: 0;
            font-family: "Raleway", sans-serif;
            font-weight: 300;
            font-size: 40px;
            color: #FFFFF;
            transition: all 0.4s ease 0s;
        }

        h1 span {
            display: block;
            font-size: 0.5em;
            line-height: 1.3;
        }

        h1 em {
            font-style: normal;
            font-weight: 600;
        }

        /* === HEADING STYLE #1 === */
        .one h1 {
            text-align: center;
            text-transform: uppercase;
            padding-bottom: 5px;
        }

        .one h1:before {
            width: 28px;
            height: 5px;
            display: block;
            content: "";
            position: absolute;
            bottom: 3px;
            left: 50%;
            margin-left: -14px;
            background-color: #FF9A5A;
        }

        .one h1:after {
            width: 100px;
            height: 1px;
            display: block;
            content: "";
            position: relative;
            margin-top: 25px;
            left: 50%;
            margin-left: -50px;
            background-color: #FF9A5A;
        }

        body {
            background: #e0e5ec;
        }

        /* CSS */
        .button-62 {
            margin-top: 30px;
            background: linear-gradient(to bottom right, #efe747, #FF9A5A);
            border: 0;
            text-align: center;
            border-radius: 12px;
            color: #FFFFFF;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: -apple-system, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 16px;
            font-weight: 500;
            line-height: 2.5;
            outline: transparent;
            padding: 0 1rem;
            text-align: center;
            text-decoration: none;
            transition: box-shadow .2s ease-in-out;
            user-select: none;
            white-space: nowrap;
        }

        .button-62:not([disabled]):focus {
            box-shadow: 0 0 .25rem rgba(0, 0, 0, 0.5), -.125rem -.125rem 1rem rgba(239, 71, 101, 0.5), .125rem .125rem 1rem rgba(255, 154, 90, 0.5);
        }

        .button-62:not([disabled]):hover {
            box-shadow: 0 0 .25rem rgba(0, 0, 0, 0.5), -.125rem -.125rem 1rem rgba(239, 71, 101, 0.5), .125rem .125rem 1rem rgba(255, 154, 90, 0.5);
        }

        .center-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="one">
        <h1>Welcome TO The Rain Prediction</h1>
    </div>
    <div class="center-container"><button class="button-62 center-container"
            onclick="redirectToPredictionInterface()">Go
            to
            Prediction
            Interface</button></div>

    <br>
    <br>
    <div class="center-container"><iframe title="Report Section" width="1024" height="600"
            src="https://app.powerbi.com/view?r=eyJrIjoiMGFmZjMyNWEtNWYyNy00YjcxLWFjZDktMjA3ZjJiODQ5ZjZlIiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9"
            frameborder="0" allowFullScreen="true"></iframe>
        <!-- Add a button to redirect to the prediction interface -->
    </div>
    <!-- JavaScript function to handle redirection -->
    <script>
        function redirectToPredictionInterface() {
            // Redirect to the prediction interface URL
            window.location.href = "http://localhost:8501/";
        }
    </script>
    """,
    unsafe_allow_html=True,
)
