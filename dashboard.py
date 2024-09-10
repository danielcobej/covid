import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import dash_html_components as html
import dash_bootstrap_components as dbc
from explainerdashboard.custom import *
from explainerdashboard import ClassifierExplainer, ExplainerDashboard, ExplainerHub





pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



# Načítame dataset do dataframu "df"
df = pd.read_excel("dataset.xlsx")

df = df.rename(columns={'Patient ID': 'Patient_ID',
                        'Patient age quantile': 'Patient_age_quantile',
                        'SARS-Cov-2 exam result': 'COVID19',
                        'Patient addmited to regular ward (1=yes, 0=no)': 'Patient_addmited_to_regular_ward',
                        'Patient addmited to semi-intensive unit (1=yes, 0=no)': 'Patient_addmited_to_semi-intensive_unit',
                        'Patient addmited to intensive care unit (1=yes, 0=no)': 'Patient_addmited_to_intensive_care_unit',
                        'Mean platelet volume ': 'Mean_platelet_volume',
                        'Red blood Cells': 'Red_blood_Cells',
                        'Mean corpuscular hemoglobin concentration (MCHC)	': 'Mean_corpuscular_hemoglobin_concentration',
                        'Mean corpuscular hemoglobin (MCH)': 'Mean_corpuscular_hemoglobin',
                        'Mean corpuscular volume (MCV)': 'Mean_corpuscular_volume',
                        'Red blood cell distribution width (RDW)': 'Red_blood_cell_distribution_width',
                        'Serum Glucose': 'Serum_Glucose',
                        'Respiratory Syncytial Virus': 'Respiratory_Syncytial_Virus',
                        'Influenza A': 'Influenza_A',
                        'Influenza B': 'Influenza_B',
                        'Parainfluenza 1': 'Parainfluenza_1',
                        'Mycoplasma pneumoniae': 'Mycoplasma_pneumoniae',
                        'Coronavirus HKU1': 'Coronavirus_HKU1',
                        'Parainfluenza 3': 'Parainfluenza_3',
                        'Chlamydophila pneumoniae': 'Chlamydophila_pneumoniae',
                        'Parainfluenza 4': 'Parainfluenza_4',
                        'Inf A H1N1 2009': 'Inf_A_H1N1_2009',
                        'Bordetella pertussis': 'Bordetella_pertussis',
                        'Parainfluenza 2': 'Parainfluenza_2',
                        'Proteina C reativa mg/dL': 'Proteina_C_reativa',
                        'Influenza B, rapid test': 'Influenza_B_rapid_test',
                        'Influenza A, rapid test': 'Influenza_A_rapid_test',
                        'Alanine transaminase': 'Alanine_transaminase',
                        'Aspartate transaminase': 'Aspartate_transaminase',
                        'Total Bilirubin': 'Total_Bilirubin',
                        'Direct Bilirubin': 'Direct_Bilirubin',
                        'Indirect Bilirubin': 'Indirect_Bilirubin',
                        'Alkaline phosphatase': 'Alkaline_phosphatase',
                        'Ionized calcium\xa0': 'Ionized_calcium',
                        'Strepto A': 'Strepto_A',
                        'pCO2 (venous blood gas analysis)': 'pCO2',
                        'Hb saturation (venous blood gas analysis)': 'Hb_saturation',
                        'Base excess (venous blood gas analysis)': 'Base_excess',
                        'pO2 (venous blood gas analysis)': 'pO2',
                        'Fio2 (venous blood gas analysis)': 'Fio2',
                        'Total CO2 (venous blood gas analysis)': 'Total_CO2',
                        'pH (venous blood gas analysis)': 'pH',
                        'HCO3 (venous blood gas analysis)': 'HCO3',
                        'Rods #': 'Rods',
                        'Urine - Esterase': 'Urine_Esterase',
                        'Urine - Aspect': 'Urine_Aspect',
                        'Urine - pH': 'Urine_pH',
                        'Urine - Hemoglobin': 'Urine_Hemoglobin',
                        'Urine - Bile pigments': 'Urine_Bile_pigments',
                        'Urine - Ketone Bodies': 'Urine_Ketone_Bodies',
                        'Urine - Nitrite': 'Urine_Nitrite',
                        'Urine - Density': 'Urine_Density',
                        'Urine - Urobilinogen': 'Urine_Urobilinogen',
                        'Urine - Protein': 'Urine_Protein',
                        'Urine - Sugar': 'Urine_Sugar',
                        'Urine - Leukocytes': 'Urine_Leukocytes',
                        'Urine - Crystals': 'Urine_Crystals',
                        'Urine - Red blood cells': 'Urine_Red_blood_cells',
                        'Urine - Hyaline cylinders': 'Urine_Hyaline_cylinders',
                        'Urine - Granular cylinders': 'Urine_Granular_cylinders',
                        'Urine - Color': 'Urine_Color',
                        'Urine - Yeasts': 'Urine_Yeasts',
                        'Partial thromboplastin time\xa0(PTT)\xa0': 'Partial_thromboplastin_time',
                        'Relationship (Patient/Normal)': 'Relationship_Patient_or_Normal',
                        'International normalized ratio (INR)': 'International_normalized_ratio',
                        'Lactic Dehydrogenase': 'Lactic_Dehydrogenase',
                        'Prothrombin time (PT), Activity': 'Prothrombin_time',
                        'Vitamin B12': 'Vitamin_B12',
                        'Creatine phosphokinase\xa0(CPK)\xa0': 'Creatine_phosphokinase',
                        'Arterial Lactic Acid': 'Arterial_Lactic_Acid',
                        'Lipase dosage': 'Lipase_dosage',
                        'D-Dimer': 'D_Dimer',
                        'Hb saturation (arterial blood gases)': 'Hb_saturation_arterial_blood_gas_analysis',
                        'pCO2 (arterial blood gas analysis)': 'pCO2_arterial_blood_gas_analysis',
                        'Base excess (arterial blood gas analysis)': 'Base_excess_arterial_blood_gas_analysis',
                        'pH (arterial blood gas analysis)': 'pH_arterial_blood_gas_analysis',
                        'Total CO2 (arterial blood gas analysis)': 'Total_CO2_arterial_blood_gas_analysis',
                        'HCO3 (arterial blood gas analysis)': 'HCO3_arterial_blood_gas_analysis',
                        'pO2 (arterial blood gas analysis)': 'pO2_arterial_blood_gas_analysis',
                        'Arteiral Fio2': 'Arteiral_Fio2',
                        'ctO2 (arterial blood gas analysis)': 'ctO2_arterial_blood_gas_analysis',
                        'Rhinovirus/Enterovirus': 'RhinovirusEnterovirus',
                        'Gamma-glutamyltransferase\xa0': 'Gamma_glutamyltransferase'
                       })

df=df.drop('Patient_ID',axis=1)

def encode_categorical_variables(dataset):
    encoding = {'positive': 1, 'negative': 0, 'not_detected': 0, 'detected': 1, 'absent': 0, 'not_done': 0, 'present': 1}
    for column_name in dataset.columns:
        if dataset[column_name].dtype == 'object':
            dataset[column_name] = dataset[column_name].replace(encoding)
            
encode_categorical_variables(df)

# Zmažeme atribúty kde je viac ako 90% NaN hodnôt

perc = 90.0
min_count =  int(((100-perc)/100)*df.shape[0] + 1)
df = df.dropna( axis=1, 
                thresh=min_count)
df.shape

# Zmažeme záznamy kde je viac ako 50% NaN hodnôt

perc = 50.0 # Here N is 75
min_count =  int(((100-perc)/100)*df.shape[1] + 1)
df = df.dropna(axis=0, 
                    thresh=min_count)

df.shape

y = df['COVID19']
X = df

X.drop('COVID19', axis=1, inplace=True)

ros = RandomOverSampler(random_state=12345)

X_resampled, y_resampled = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=321)

XGB = XGBClassifier()
XGB.fit(X_train, y_train)
y_pred_XGB = XGB.predict(X_test)

Accuracy_XGB = metrics.accuracy_score(y_test, y_pred_XGB)
Precision_XGB = metrics.precision_score(y_test, y_pred_XGB)
Sensitivity_recall_XGB = metrics.recall_score(y_test, y_pred_XGB)
Specificity_XGB = metrics.recall_score(y_test, y_pred_XGB, pos_label=0)
F1_score_XGB = metrics.f1_score(y_test, y_pred_XGB)
print({"Accuracy":Accuracy_XGB})
print({"Precision":Precision_XGB})
print({"Sensitivity":Sensitivity_recall_XGB})
print({"Specificity":Specificity_XGB})
print({"F1_score":F1_score_XGB})

explainer = ClassifierExplainer(XGB, X_test, y_test)


class HomeTab(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Domov")

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.H5("Názov práce"),
                    html.Div("Podpora diagnostiky COVID-19 pomocou vhodných metód strojového učenia"),
                    html.H5("Autor"),
                    html.Div("Daniel Čobej"),
                    html.H5("Vedúci práce"),
                    html.Div("prof. Ing. Ján Paralič, PhD."),
                    html.H5("Konzultant"),
                    html.Div("Ing. Oliver Lohaj"),
                    html.H5("Cieľ aplikácie"),
                    html.Div("Cieľom tejto práce bolo vytvoriť klasifikačný model na podporu diagnostiky ochorenia COVID-19."),
                    html.Div("Vo fáze modelovania bolo vytvorených 20 modelov. V tejto webovej aplikácii sa snažíme zamerať na vysvetliteľnosť najlepšieho modelu ktorý bol vytvorený pomocou klasifikačného algoritmu XGBoost."),
                    html.Div("Aplikácia obsahuje okrem domovskej stránky ďalšie dve karty určené na vysvetlenie samotného modelu a vysvetlenie klasifikácie pri jednotlivých pacientoch")
                    ])
                ])
            ])


class VysvetlenieModelu(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Vysvetlenie modelu")
        self.shap_summary = ShapSummaryComponent(explainer,
                                hide_title=True, hide_selector=True,
                                hide_depth=True, depth=15,
                                hide_cats=True, cats=True, hide_type=True, hide_popout=True, )
        
        self.auc = ClassifierModelSummaryComponent(explainer,
                                                   title='Výkonnosť modelu', name=None, hide_title=False, 
                                                   hide_subtitle=False, hide_footer=True, hide_cutoff=True, hide_selector=False, 
                                                   pos_label=None, cutoff=0.5, round=3, show_metrics=None, description=None)
        
        self.roc_auc = RocAucComponent(explainer, hide_title=True, hide_footer=True, hide_cutoff=True, hide_selector=True)

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H5("Atribúty a ich popis"),
                    html.Table([
                        html.Thead(
                            html.Tr([html.Th("Atribút", style={"border-right": "1px solid black"}), html.Th("Popis")]), style={"border": "1px solid black"}
                        ),
                        html.Tbody([
                            html.Tr([html.Td("Patient_age_quantile", style={"border-right": "1px solid black"}), html.Td("Veková kvantila pacienta")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Patient_addmited_to_regular_ward", style={"border-right": "1px solid black"}), html.Td("Pacient prijatý na oddelenie")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Patient_addmited_to_semi-intensive_unit", style={"border-right": "1px solid black"}), html.Td("Pacient prijatý na polointenzívnu jednotku")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Patient_addmited_to_intensive_care_unit", style={"border-right": "1px solid black"}), html.Td("Pacient prijatý na jednotku intenzívnej starostlivosti")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Hematocrit", style={"border-right": "1px solid black"}), html.Td("Hematokrit")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Hemoglobin", style={"border-right": "1px solid black"}), html.Td("Hemoglobín")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Platelets", style={"border-right": "1px solid black"}), html.Td("Trombocyty")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Mean_platelet_volume", style={"border-right": "1px solid black"}), html.Td("Priemerný objem trombocytov")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Red_blood_Cells", style={"border-right": "1px solid black"}), html.Td("Červené krvné bunky")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Lymphocytes", style={"border-right": "1px solid black"}), html.Td("Lymfocyty")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Mean corpuscular hemoglobin concentration (MCHC)", style={"border-right": "1px solid black"}), html.Td("Priemerná koncentrácia hemoglobínu v erytrocytoch")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Leukocytes", style={"border-right": "1px solid black"}), html.Td("Leukocyty")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Basophils", style={"border-right": "1px solid black"}), html.Td("Bazofily")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Mean_corpuscular_hemoglobin", style={"border-right": "1px solid black"}), html.Td("Priemerný obsah hemoglobínu v erytrocytoch")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Eosinophils", style={"border-right": "1px solid black"}), html.Td("Eozinofily")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Mean_corpuscular_volume", style={"border-right": "1px solid black"}), html.Td("Priemerný objem erytrocytov")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Monocytes", style={"border-right": "1px solid black"}), html.Td("Monocyty")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Red_blood_cell_distribution_width", style={"border-right": "1px solid black"}), html.Td("Rozptyl červených krvných buniek")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Respiratory_Syncytial_Virus", style={"border-right": "1px solid black"}), html.Td("Respiračný syncyciálny vírus")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Influenza_A", style={"border-right": "1px solid black"}), html.Td("Chrípka A")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Influenza_B", style={"border-right": "1px solid black"}), html.Td("Chrípka B")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Parainfluenza_1", style={"border-right": "1px solid black"}), html.Td("Parainfluenza 1")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("CoronavirusNL63", style={"border-right": "1px solid black"}), html.Td("Koronavírus NL63")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("RhinovirusEnterovirus", style={"border-right": "1px solid black"}), html.Td("Rhinovírus/Enterovírus")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Coronavirus_HKU1", style={"border-right": "1px solid black"}), html.Td("Koronavírus HKU1")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Parainfluenza_3", style={"border-right": "1px solid black"}), html.Td("Parainfluenza 3")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Chlamydophila_pneumoniae", style={"border-right": "1px solid black"}), html.Td("Chlamydophila pneumoniae")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Adenovirus", style={"border-right": "1px solid black"}), html.Td("Adenovírus")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Parainfluenza_4", style={"border-right": "1px solid black"}), html.Td("Parainfluenza 4")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Coronavirus229E", style={"border-right": "1px solid black"}), html.Td("Koronavírus 229E")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("CoronavirusOC43", style={"border-right": "1px solid black"}), html.Td("Koronavírus OC43")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Inf_A_H1N1_2009", style={"border-right": "1px solid black"}), html.Td("Chrípka A H1N1 2009")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Bordetella_pertussis", style={"border-right": "1px solid black"}), html.Td("Bordetella pertussis")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Metapneumovirus", style={"border-right": "1px solid black"}), html.Td("Metapneumovírus")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Parainfluenza_2", style={"border-right": "1px solid black"}), html.Td("Parainfluenza 2")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Influenza_B_rapid_test", style={"border-right": "1px solid black"}), html.Td("Rýchly test na chrípku B")], style={"border": "1px solid black"}),
                                html.Tr([html.Td("Influenza_A_rapid_test", style={"border-right": "1px solid black"}), html.Td("Rýchly test na chrípku A")], style={"border": "1px solid black"})
                        ], style={"border": "1px solid black"})
                    ], style={"border-collapse": "collapse", "border": "1px solid black", "width": "100%"})
                ])
            ]),
            html.Br(), # Medzera medzi riadkami
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.H5("Výkonnosť modelu"),
                    html.Div("Detaily výkonnosti modelu môžeme vidieť na tabuľke na pravej strane."),
                    html.Br(),
                    html.H5("Specificita: 0.821"),
                    html.H5("Senzitivita: 0.958"),
                ], width=6),
                dbc.Col([
                    self.auc.layout(),
                ], width=6)
            ]),
            html.Br(), # Medzera medzi riadkami
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.H5("Najvplyvnejšie atribúty na klasifikáciu"),
                    html.Div("Na pravej strane môžeme vidieť jednotlivé atribúty a ich vplyv pri klasifikácii modelu. Atribúty sú zoradené od najdôležitejších po najmenej dôležité."),
                ], width=6),
                dbc.Col([
                    self.shap_summary.layout(),
                ], width=6),
            ]),
            html.Br(), # Medzera medzi riadkami
            dbc.Row([
                dbc.Col([
                    html.H5("ROC krivka a AUC metrika"),
                    html.Div("ROC krivka ilustruje schopnosť modelu odlišovať medzi pozitívnymi a negatívnymi triedami. AUC metrika vyjadruje plochu pod ROC krivkou."),
                ], width=6),
                dbc.Col([
                    self.roc_auc.layout(),
                ], width=6)
            ]),
        ])





#---------------------------------------------------------------------------------------------------------------
class CustomPredictionsTab(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Predikcia")

        self.index = ClassifierRandomIndexComponent(explainer,
                                                    hide_title=True, hide_index=False,
                                                    hide_slider=True, hide_labels=False,
                                                    hide_pred_or_perc=True,
                                                    hide_selector=True, hide_button=False, index_dropdown=True)

        self.contributions = ShapContributionsTableComponent(explainer,
                                                            hide_title=True, hide_index=True,
                                                            hide_depth=True, hide_sort=True,
                                                            hide_orientation=True, hide_cats=True,
                                                            hide_selector=True,
                                                            sort='high-to-low')
        self.contributions2 = ShapContributionsGraphComponent(explainer,
                                                            hide_title=True, hide_index=True,
                                                            hide_depth=True, hide_sort=True,
                                                            hide_orientation=True, hide_cats=True,
                                                            hide_selector=True, hide_popout=True,
                                                            sort='high-to-low')
        self.summary = ClassifierPredictionSummaryComponent(explainer,
                                                            hide_title=True, hide_index=True, hide_selector=True)

        self.connector = IndexConnector(self.index, [self.contributions, self.summary, self.contributions2])

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H4("Výber náhodného pacienta"),
                    self.index.layout()
                ],width=4),
                dbc.Col([
                    html.H4("Výsledok klasifikácie (* je správny výsledok, % sú predpoveď modelu)"),
                    self.summary.layout()
                ]),

            ]),
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.Br(),
                    html.H3("ZELENÁ farba vyznačuje atribúty ktoré predpovedajú POZITÍVNY COVID19"),
                    html.H3("ČERVENÁ farba vyznačuje atribúty ktoré predpovedajú NEGATÍVNY COVID19"),
                    html.H4("Do úvahy sa berie nie len atribút ale aj jeho hodnota pri konkrétnom pacientovi"),
                    self.contributions2.layout()
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.contributions.layout()
                ]),
            ])
        ])

ExplainerDashboard(explainer, [HomeTab, VysvetlenieModelu, CustomPredictionsTab], hide_header=True, hide_poweredby=True).run()