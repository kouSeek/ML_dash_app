import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output
import runModel as ml

hotels = ['Americas Best Value Inn & Suites Bismarck', 'Americas Best Value Inn & Suites Melbourne', 'Americas Best Value Inn Charlotte, NC','Americas Best Value Inn & Suites Boise',  "Americas Best Value Inn Midlothian Cedar Hill", "Red Lion Inn & Suites Modesto", "Knights Inn Virginia Beach on 29th St", 'Red Lion Hotel on the River Jantzen Beach', 'Americas Best Value Inn Scarborough Portland', 'Americas Best Value Inn Nashville Downtown', 'Americas Best Value Inn Nashville Airport S', 'Americas Best Value Inn Lake Tahoe-Tahoe City']

test_sizes = [30, 45, 60, 90, 15, 7]
models = [ml.XGBRegressor()]
weights = [1]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
app.title = "Demand Forecasting, Koushik"

app.layout = html.Div([
	html.Div([
		html.Label("Select Property:"),
		dcc.Dropdown(options=[{'label':i, 'value':i} for i in hotels],
			value=hotels[0],
			id="my_dropdown",
		),
	], style={'width': '30%', 'display': 'inline-block'}
	),

	html.Div([
		html.Label("Select Confidence level:"),
		dcc.Dropdown(options=[
			{"label":"50%", "value":0.5},
			{"label":"40%", "value":0.4},
			{"label":"30%", "value":0.3},
			{"label":"25%", "value":0.25},
			{"label":"20%", "value":0.2},
			{"label":"10%", "value":0.1},
			],
			value=0.5,
			id="my_conf",
		),
	], style={'width': '15%', 'display': 'inline-block'}
	),

	html.Div([
		html.Label("Select Prediction days:"),
		dcc.Dropdown(options=[{'label':i, 'value':i} for i in test_sizes],
			value=test_sizes[0],
			id="my_test_size",
		),
	], style={'width': '15%', 'display': 'inline-block', }
	),	

	dcc.Tabs([
		dcc.Tab(label='Property Details', 
			children=[dcc.Markdown(id="div_eda", style={'height': 200, 'width': '17%', 'display': 'inline-block', "vertical-align":"top",}),
					dcc.Graph(id="div_history", style={'height': 600, "width": 1000, 'display': 'inline-block'})]
		),
		dcc.Tab(label='Variable Impact',
			children=[dcc.Graph(id="div_feat_imp", style={'height': 600, "width": 1300,}, )]
		),
		dcc.Tab(label='Prediction Plot', 
			children=[dcc.Markdown(id="div_metrics", style={'width': '20%', 'display': 'inline-block', "vertical-align":"top",}),
					dcc.Graph(id="div_prediction", style={'height': 600, "width": 1000, 'display': 'inline-block'})]
		),
		dcc.Tab(label="Prediction Data",
			children=[dash_table.DataTable(
				id="table_pred", page_size = 15, style_header={'fontWeight': 'bold'},
			)]
		),
		dcc.Tab(label="Raw Data",
			children=[dash_table.DataTable(
				id="table", page_size = 15, style_header={'fontWeight': 'bold'},
			)]
		),
	],
	colors={
        "border": "yellow",
        "primary": "red",
        "background": '#21bDFF'
    	}
	),

],
)


@app.callback(
	[Output(component_id="div_eda", component_property="children"),
	Output(component_id="div_history", component_property="figure"),
	Output(component_id="div_feat_imp", component_property="figure"),
	Output(component_id="div_prediction", component_property="figure"),
	Output(component_id="div_metrics", component_property="children"),
	Output(component_id="table", component_property="data"),
	Output(component_id="table", component_property="columns"),
	Output(component_id="table_pred", component_property="data"),
	Output(component_id="table_pred", component_property="columns"),
	],
	[Input(component_id="my_dropdown", component_property="value"),
	Input(component_id="my_conf", component_property="value"),
	Input(component_id="my_test_size", component_property="value"),
	]
)
def update_div_prediction(prop_name, confidence, test_size):
	df = ml.prepData(prop_name)
	eda = str(ml.getEDA(prop_name))
	results, y_test, y_pred, ts, var_imp = ml.runModel(models, weights, df, test_size, confidence)

	plot_history = dict(
		data=[dict(x=df.ArrivalDate, y=df.Count, name="Prediction", ),
		]
	)
	plot_pred= dict(
		data=[dict(x=ts, y=y_pred, name="Prediction", line=dict(color='purple') ),
			dict(x=ts, y=y_test, name="Actual", line=dict(color='green') ),
			dict(x=ts, y=y_test+y_test.mean()*confidence, name="Actual+"+str(confidence*100)+"%", line=dict(color='lime', dash='dash') ),
			dict(x=ts, y=y_test-y_test.mean()*confidence, name="Actual-"+str(confidence*100)+"%", line=dict(color='lime', dash='dash') ),
		]
	)
	plot_feat_imp = dict(
		data=[dict(x=list(var_imp.keys()), y=list(var_imp.values()), name="Variable Importance", type= 'bar'),
		]
	)
	df.to_csv(prop_name+"_temp.csv", index=False)
	data = df.to_dict('records')
	cols = [{"name": i, "id": i} for i in df.columns]
	
	pred_cols = ["Actual", "predicted", f"Actual+{confidence*100}%", f"Actual-{confidence*100}%"]
	pred_data = pd.DataFrame({pred_cols[0]:y_test, pred_cols[1]:y_pred, pred_cols[2]:y_test*(1+confidence), pred_cols[3]:y_test*(1-confidence)}).to_dict('records')
	pred_columns=[{"name": i, "id": i} for i in pred_cols]
	return eda, plot_history, plot_feat_imp, plot_pred, results, data, cols, pred_data, pred_columns



if __name__ == "__main__":
	app.run_server(host="0.0.0.0",debug=True, dev_tools_ui=False)