MODEL_DIR = '../models'
MODEL_FILE = 'ensemble_models.pkl'
FEATURE_IMP_FILE = 'ensemble_models_feature_importances.pkl'

DATA_PATH = '../data/processed/02_cleaned_df.pkl'

ROLE_COLS = ['DevType']
TECH_COLS = ['LanguageHaveWorkedWith',
             'DatabaseHaveWorkedWith',
             'WebframeHaveWorkedWith',
             'MiscTechHaveWorkedWith',
             'ToolsTechHaveWorkedWith']

TECH_NAMES = {'LanguageHaveWorkedWith': "Languages",
              'DatabaseHaveWorkedWith': "Databases",
              'WebframeHaveWorkedWith': "Web Frameworks",
              'MiscTechHaveWorkedWith': "Other Tech",
              'ToolsTechHaveWorkedWith': "Tools"}

# -----------

# *** Load libraries ***

from scripts.SkillsJobModel import SkillsJobModel
from scripts.preprocessing import one_hot_encode

import os
import pickle

import pandas as pd
import plotly.express as px

import dash
import dash_bootstrap_components as dbc

from dash import dcc
from dash.dependencies import Output, Input, State
from dash import html, callback_context

# -----------

# *** Initialize ***

# .. Initialize App
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP,
                                      dbc.icons.BOOTSTRAP,
                                      dbc.icons.FONT_AWESOME])

# .. Initialize App
## Load skills from original dataframe to get groupings (Languages, Databases, ...)
df = pd.read_pickle(DATA_PATH)
skills_groups = {col: one_hot_encode(df, [col]).columns.get_level_values(1).tolist()
                 for col in TECH_COLS}

### --- Load model ----
skills_jobs = SkillsJobModel(os.path.join(MODEL_DIR, MODEL_FILE))


# ************************************************
#                    UI
# ************************************************

heading_div = html.Div(
    [
        html.H1("Develop your career in Tech"),
        dcc.Markdown("*****")
    ]
)

### **** Skills UI ****

skills_checklists = [
    html.Div(
        [dcc.Markdown(
            TECH_NAMES[group_name],
            className='skill_group_title'),
            dbc.Checklist(
                id=group_name + "_checklist",
                options=[{'label': skill, 'value': skill}
                         for skill in group_skills],
                value=[],
                inline=True,
                className='skill_group'
            )]
    )
    for group_name, group_skills in skills_groups.items()
]

checklists_ids = [checklist.children[1].id
                  for checklist in skills_checklists]

skills_div = html.Div(
    [
        dcc.Markdown("### Your skills:"),
        html.Div(skills_checklists)
    ],
    id='skills_div', className='control_card')

### **** Match UI ****

predictions_div = html.Div(
    [
        dcc.Markdown("## What's my current match?"),
        dbc.Button("Find out", id='match_button',
                   size="lg", className="text-center mb-3 "),
        dcc.Loading(dcc.Graph(id='predictions_plot'))
    ],
    id='match_div'
)


# ************************************************
#                  Callbacks
# ************************************************

def extract_skill_list_from_div(skills_div):
    # Extract selected skills
    # ... Unpack values from parent div
    skills_subdiv = skills_div[1]['props']['children']
    selected_skills = [skill_group['props']['children'][1]['props']['value']
                       for skill_group in skills_subdiv]
    # ... Flatten nested lists
    selected_skills = sum(selected_skills, [])
    return selected_skills


@app.callback(Output('predictions_plot', 'figure'),
              [Input('match_button', 'n_clicks')],
              State('skills_div', 'children'))
def update_prediction_plot(n_clicks, skills_div):
    # Get selected skills
    selected_skills = extract_skill_list_from_div(skills_div)
    if len(selected_skills) == 0:
        return px.bar(orientation='h', width=10, height=10)

    # Predict values
    # predictions = generate_random_predictions(model)
    predictions = skills_jobs.predict_jobs_probs(selected_skills)

    # Plot and return
    fig = px.bar(predictions.sort_values(),
                 orientation='h',
                 width=1200, height=500)

    fig \
        .update_xaxes(title='',
                      visible=True,
                      tickformat=',.0%',
                      range=[-0.05, 1]) \
        .update_yaxes(title='',
                      visible=True) \
        .update_layout(showlegend=False,
                       font=dict(size=16),
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')

    return fig



# ************************************************
#                Compile and run
# ************************************************

app.layout = dbc.Container(
    [
        # Storage
        dcc.Store(id='store_selected_job'),

        # UI
        dbc.Row(
            [heading_div], align="top", className="heading",
        ),
        dbc.Row(
            [dbc.Col(skills_div)], align="top", className="controls",
        ),

        dbc.Row(
            [dbc.Col(predictions_div)], align="center", className="output"
        )

    ],
    fluid=True,
    className="page"
)

if __name__ == '__main__':
    app.run_server(debug=True, port='8051')
