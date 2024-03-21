# importeren van benodigde pakkages
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import folium
import streamlit as st
import seaborn as sns

import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Dataset 1 inladen
df = pd.read_csv("schedule_airport.csv")
# Kolom inkomend gesplitst in inkomende en uitgaande vluchten voor eventueel later gebruik. 
df["inkomend"] = df["LSV"].apply(lambda x: 1 if x == "L" else 0)
df.drop(['LSV','DL1', 'IX1', 'DL2', 'IX2', "Identifier"], axis = 1, inplace = True)
df.columns = ['datum', 'vluchtnummer', 'tijd_gepland', 'tijd', 'gate_gepland', 'gate', 'vliegtuigtype',
       'landingsbaan', 'RWC', 'bestemming', 'aankomend']

df["gate_gepland"] = np.where(df["gate_gepland"] == "-", np.nan, df["gate_gepland"])
df["gate"] = np.where(df["gate"] == "-", np.nan, df["gate"])
df["RWC"] = np.where(df["RWC"] == "-", np.nan, df["RWC"])

# Dataset 2 inladen
df2 = pd.read_csv("airports-extended-clean.csv", sep=";")
df2 = df2[df2["Source"] == "OurAirports"]
df2 = df2[df2["Type"] == "airport"]

df2.replace(r'\N', np.nan, inplace=True)
df2.drop(['Airport ID', 'DST','Tz', 'Type', 'Source'], axis = 1, inplace = True)
df2.columns = ['naam', 'stad', 'land', 'IATA', 'ICAO', 'lat', 'lng',
       'hoogte_vliegveld', 'tijdzone']

df2['lat'] = df2['lat'].str.replace(',', '.')
df2['lat'] = df2['lat'].astype(float)
 
df2['lng'] = df2['lng'].str.replace(',', '.')
df2['lng'] = df2['lng'].astype(float)
 
df2['tijdzone'] = df2['tijdzone'].str.replace(',', '.')
df2['tijdzone'] = df2['tijdzone'].astype(float)

# Nieuw dataframe aangemaakt voor de plot waarbij is ingezoomd op 3 landen.
df_zoom = df2[(df2["land"] == "Netherlands") | (df2["land"] == "Germany") | (df2["land"] == "United Kingdom")]

# Inhoudsopgave
st.sidebar.title('Inhoudsopgave')
option = st.sidebar.radio("Selecteer uw keuze:", ['Inleiding','Wereldkaart', 'Ingezoomd op drie landen', 'Lijndiagram', 'Voorspellingsmodel'])
custom_palette = sns.color_palette('GnBu') 

if option == 'Inleiding':
    # Pagina-titel en introductie
    st.title("Vluchtgegevens van en naar Zürich in 2019 en 2020")

    st.write(
        "Welkom bij ons interactieve dashboard, waarbij we jullie meennemen door onze "
        "datasets. In dit dashboard is informatie te vinden over de vluchten van en naar Zürich "
        "Airport, Zwitserland. Er zijn een aantal kaarten weergegeven en er is een voorspellingsmodel "
        "gemaakt die de vertraging van vluchten voorspelt.")

    # Inhoud van het dashboard.
    st.write("### Inhoud van het dashboard:")
    st.write(
        "1. Inleiding: korte introductie van de dataset."
    )
    st.write(
        "2. Wereldkaart: Inzicht van het aantal vliegvelden over de wereld."
    )
    st.write(
        "3. Ingezoomd op drie landen: Inzicht van het aantal vliegvelden in Nederland, Duitsland en het Verenigd Koninkrijk."
    )
    st.write(
        "4. Lijndiagram: Weergave van het aantal vliegtuigen aan de grond per vliegveld."
    )
    st.write(
        "5. Voorspellingsmodel: Een Machine learning model dat een voorspelling maakt van de vertraging van vluchten."
        
    )
    
    # Bronvermelding
    st.write(
        "De datasets over de vluchtgegevens en de gegevens van vliegvelden zijn afkomstig van Kaggle."
    )

    # Ons zelf voorstellen
    st.write(
        "Deze blog is gemaakt door team 8 van de minor Data Science aan de Hogeschool van Amsterdam. Dit team bestaat uit: Oumaima Chaara, Alysha Mannaart en Esmee Springer."
        
    )
    
    
elif option == 'Wereldkaart':
    st.title("Wereldkaart")
    tabs = st.tabs(["Over de wereld", "Ingezoomd op drie landen"])

    # Tab voor wereldkaart
    with tabs[0]:
        
        # streamlit titel
        st.header('Ligging van aantal vliegvelden')
        st.write('Aantal vliegvelden over de wereld')

        # Maak een lege kaart
        m = folium.Map(location=[0, 0], zoom_start=2) 
        marker_cluster = MarkerCluster().add_to(m)
        vliegvelden_per_land = df2['land'].value_counts().to_dict()

        # Definieer drempelwaarden voor concentratiecategorieën op basis van het aantal vliegvelden per land
        max_count = max(vliegvelden_per_land.values())
        min_count = min(vliegvelden_per_land.values())
        threshold_low = min_count + (max_count - min_count) / 3
        threshold_high = min_count + 2 * (max_count - min_count) / 3

        color_palette = {
            'Low': 'green',
            'Medium': 'yellow',
            'High': 'red'
        }

        for index, row in df2.iterrows():
            popup_text = f"Naam: {row['naam']}<br>Stad: {row['stad']}<br>Land: {row['land']}"
            if vliegvelden_per_land[row['land']] < threshold_low:
                concentration = 'Low'
            elif vliegvelden_per_land[row['land']] < threshold_high:
                concentration = 'Medium'
            else:
                concentration = 'High'
            color = color_palette[concentration]
            folium.Marker(location=[row['lat'], row['lng']], popup=popup_text,icon=folium.Icon(icon='plane', prefix='fa')).add_to(marker_cluster)

        folium.plugins.Fullscreen().add_to(m)

        # Toon de kaart
        st_data = st_folium(m, width = 725)
        
    # Tab voor kaart ingezoomd  op drie landen    
    with tabs[1]:

        # Streamlit titel
        st.header('Ligging van aantal vliegvelden in de drie gekozen landen')
        st.write('Aantal vliegvelden in Nederland, Duitsland en het Verenigd Koninkrijk.')

        # Maak een lege kaart
        m = folium.Map(location=[52.1326,5.2913], zoom_start=5) 
        marker_cluster = MarkerCluster().add_to(m)
        vliegvelden_per_land = df_zoom['land'].value_counts().to_dict()

        # Definieer drempelwaarden voor concentratiecategorieën op basis van het aantal vliegvelden per land
        max_count = max(vliegvelden_per_land.values())
        min_count = min(vliegvelden_per_land.values())
        threshold_low = min_count + (max_count - min_count) / 3
        threshold_high = min_count + 2 * (max_count - min_count) / 3

        color_palette = {
            'Low': 'green',
            'Medium': 'yellow',
            'High': 'red'
        }

        for index, row in df_zoom.iterrows():
            popup_text = f"Naam: {row['naam']}<br>Stad: {row['stad']}<br>Land: {row['land']}"
            if vliegvelden_per_land[row['land']] < threshold_low:
                concentration = 'Low'
            elif vliegvelden_per_land[row['land']] < threshold_high:
                concentration = 'Medium'
            else:
                concentration = 'High'
            color = color_palette[concentration]
            folium.Marker(location=[row['lat'], row['lng']], popup=popup_text,icon=folium.Icon(icon='plane', prefix='fa')).add_to(marker_cluster)

        folium.plugins.Fullscreen().add_to(m)

        # Toon de kaart
        st_data = st_folium(m, width = 725)
        
        
elif option == 'Ingezoomd op drie landen':
    st.title("Ingezoomd op drie landen")
    tabs = st.tabs(["Ongefilterd", "Gefilterd", "Vluchtgegevens bekend"])

    # Tab voor drie landen ongefilterd
    with tabs[0]:
        
        # Streamlit titel
        st.header('Weergave van het aantal vliegvelden in Nederland, Duitsland en het Verenigd Koninkrijk')
        st.write('Data is nog ongefilterd.')

        # Maak een lege kaart 
        m = folium.Map(location=[52.1326, 5.2913], tiles="OpenStreetMap", zoom_start=5)
        
        # Legenda in html
        legend_html = '''
        <div style="position: fixed; 
             bottom: 50px; left: 50px; width: 120px; height: 80px; 
             border:2px solid grey; z-index:9999; font-size:9px;
             background-color: white;
             opacity: 0.8;
             ">
        <b>Vliegvelden in:</b> <br>
        <svg height="10" width="10"><circle cx="5" cy="5" r="4" fill="red" /></svg> Duitsland <br>
        <svg height="10" width="10"><circle cx="5" cy="5" r="4" fill="orange" /></svg> Nederland <br>
        <svg height="10" width="10"><circle cx="5" cy="5" r="4" fill="blue" /></svg> Verenigd Koninkrijk <br>
        </div>
        '''
        
        # Voeg de vliegvelden per land toe als aparte FeatureGroup
        group_netherlands = folium.FeatureGroup(name="Netherlands").add_to(m)
        group_germany = folium.FeatureGroup(name="Germany").add_to(m)
        group_uk = folium.FeatureGroup(name="United Kingdom").add_to(m)
        
        for i in df_zoom.index:
            if df_zoom.loc[i, 'land'] == 'Netherlands':
                folium.CircleMarker(
                    location=[df_zoom.loc[i, "lat"], df_zoom.loc[i, "lng"]],
                    tooltip=df_zoom.loc[i, 'naam'],
                    radius=5,
                    fill=True,
                    color='orange',
                    fill_opacity=0.6
                ).add_to(group_netherlands)
            elif df_zoom.loc[i, 'land'] == 'Germany':
                folium.CircleMarker(
                    location=[df_zoom.loc[i, "lat"], df_zoom.loc[i, "lng"]],
                    tooltip=df_zoom.loc[i, 'naam'],
                    radius=5,
                    fill=True,
                    color='red',
                    fill_opacity=0.6
                ).add_to(group_germany)
            elif df_zoom.loc[i, 'land'] == 'United Kingdom':
                folium.CircleMarker(
                    location=[df_zoom.loc[i, "lat"], df_zoom.loc[i, "lng"]],
                    tooltip=df_zoom.loc[i, 'naam'],
                    radius=5,
                    fill=True,
                    color='blue',
                    fill_opacity=0.6
                ).add_to(group_uk)

        # Voeg een legenda toe
        folium.map.LayerControl('topright', collapsed=False).add_to(m)
        m.get_root().html.add_child(folium.Element(legend_html))
   
        # Toon de kaart
        st_data = st_folium(m, width = 700, height = 500)
        
    # Tab voor 3 landen gefilterd
    with tabs[1]:
        
        df_zoom = df_zoom[df_zoom["naam"]!="Hunt Field"]
        df_zoom = df_zoom[df_zoom["naam"]!="Whiting Field Naval Air Station South Airport"]
        df_zoom = df_zoom[df_zoom["naam"]!="Emporia Municipal Airport"]
        df_zoom = df_zoom[df_zoom["naam"]!="Engels heliport"]
        df_zoom = df_zoom[df_zoom["naam"]!="Orlampa Inc Airport"]
        df_zoom = df_zoom[df_zoom["naam"]!="Skå-Edeby Airport"]   

        # Streamlit titel
        st.header('Weergave van het aantal vliegvelden in Nederland, Duitsland en het Verenigd Koninkrijk')
        st.write('Data is gefilterd.')

        # Maak een lege kaart
        m = folium.Map(location=[52.1326, 5.2913], tiles="OpenStreetMap", zoom_start=5)
        legend_html = '''
        <div style="position: fixed; 
             bottom: 50px; left: 50px; width: 120px; height: 80; 
             border:2px solid grey; z-index:9999; font-size:9px;
             background-color: white;
             opacity: 0.8;
             ">
        <b>Vliegvelden in:</b> <br>
        <svg height="10" width="10"><circle cx="5" cy="5" r="4" fill="red" /></svg> Duitsland <br>
        <svg height="10" width="10"><circle cx="5" cy="5" r="4" fill="orange" /></svg> Nederland <br>
        <svg height="10" width="10"><circle cx="5" cy="5" r="4" fill="blue" /></svg> Verenigd Koninkrijk <br>
        </div>
        '''

        # Voeg de vliegvelden per land toe als aparte FeatureGroup
        group_netherlands = folium.FeatureGroup(name="Netherlands").add_to(m)
        group_germany = folium.FeatureGroup(name="Germany").add_to(m)
        group_uk = folium.FeatureGroup(name="United Kingdom").add_to(m)
        
        for i in df_zoom.index:
            if df_zoom.loc[i, 'land'] == 'Netherlands':
                folium.CircleMarker(
                    location=[df_zoom.loc[i, "lat"], df_zoom.loc[i, "lng"]],
                    tooltip=df_zoom.loc[i, 'naam'],
                    radius=5,
                    fill=True,
                    color='orange',
                    fill_opacity=0.6
                ).add_to(group_netherlands)
            elif df_zoom.loc[i, 'land'] == 'Germany':
                folium.CircleMarker(
                    location=[df_zoom.loc[i, "lat"], df_zoom.loc[i, "lng"]],
                    tooltip=df_zoom.loc[i, 'naam'],
                    radius=5,
                    fill=True,
                    color='red',
                    fill_opacity=0.6
                ).add_to(group_germany)
            elif df_zoom.loc[i, 'land'] == 'United Kingdom':
                folium.CircleMarker(
                    location=[df_zoom.loc[i, "lat"], df_zoom.loc[i, "lng"]],
                    tooltip=df_zoom.loc[i, 'naam'],
                    radius=5,
                    fill=True,
                    color='blue',
                    fill_opacity=0.6
                ).add_to(group_uk)

        # Voeg een legenda toe
        folium.map.LayerControl('topright', collapsed=False).add_to(m)
        m.get_root().html.add_child(folium.Element(legend_html))

        # Toon de kaart
        st_data = st_folium(m, width = 700, height = 500)
        
    # Tab voor drie landen waarvan vluchtgegevens bekend zijn.   
    with tabs[2]:
        
        # De twee datasets mergen op de overeenkomstige kolom. 
        merged_df = pd.merge(df, df2, left_on = "bestemming", right_on = "ICAO", how = "inner")
        vlucht_bekend = merged_df[merged_df['land'].isin(df_zoom["land"])]
        uniek = vlucht_bekend["ICAO"].unique()
        df_uniek = vlucht_bekend.drop_duplicates(subset=['ICAO'])

        # Lijst 1 staat voor dat er vluchtgegevens bekend zijn, lijst 2 staat voor dat er geen vluchtgegevens bekend zijn.
        lijst1 = []
        lijst2 = []
        for i in df_zoom.index:
            if df_zoom.loc[i, "ICAO"] in uniek:
                lijst1.append(df_zoom.loc[i])
            else:
                lijst2.append(df_zoom.loc[i])

        lijst1_df = pd.DataFrame(lijst1)
        lijst2_df = pd.DataFrame(lijst2)
        
        # Streamlit titel
        st.header('Weergave van het aantal vliegvelden in Nederland, Duitsland en het Verenigd Koninkrijk')
        st.write('Data waarvan de vluchtgegevens bekend zijn.')

        # Maak een lege kaart aan
        m = folium.Map(location=[52.1326, 5.2913], tiles="OpenStreetMap", zoom_start=5)
        legend_html = '''
        <div style="position: fixed; 
             bottom: 50px; left: 50px; width: 120px; height: 80; 
             border:2px solid grey; z-index:9999; font-size:9px;
             background-color: white;
             opacity: 0.8;
             ">
        <b>Vliegvelden in:</b> <br>
        <svg height="10" width="10"><circle cx="5" cy="5" r="4" fill="red" /></svg> Duitsland <br>
        <svg height="10" width="10"><circle cx="5" cy="5" r="4" fill="orange" /></svg> Nederland <br>
        <svg height="10" width="10"><circle cx="5" cy="5" r="4" fill="blue" /></svg> Verenigd Koninkrijk <br>
        <svg height="10" width="10"><circle cx="5" cy="5" r="4" fill="black" /></svg> Geen vluchtgegevens <br>
        </div>
        '''

        # Voeg de vliegvelden per land toe als aparte FeatureGroup
        group_netherlands = folium.FeatureGroup(name="Nederland").add_to(m)
        group_germany = folium.FeatureGroup(name="Duitsland").add_to(m)
        group_uk = folium.FeatureGroup(name="Verenigd Koninkrijk").add_to(m)
        
        for i in lijst1_df.index:
            if lijst1_df.loc[i, 'land'] == 'Netherlands':
                folium.CircleMarker(
                    location=[lijst1_df.loc[i, "lat"], lijst1_df.loc[i, "lng"]],
                    tooltip=lijst1_df.loc[i, 'naam'],
                    radius=5,
                    fill=True,
                    color='orange',
                    fill_opacity=0.6
                ).add_to(group_netherlands)
            elif lijst1_df.loc[i, 'land'] == 'Germany':
                folium.CircleMarker(
                    location=[lijst1_df.loc[i, "lat"], lijst1_df.loc[i, "lng"]],
                    tooltip=lijst1_df.loc[i, 'naam'],
                    radius=5,
                    fill=True,
                    color='red',
                    fill_opacity=0.6
                ).add_to(group_germany)
            elif lijst1_df.loc[i, 'land'] == 'United Kingdom':
                folium.CircleMarker(
                    location=[lijst1_df.loc[i, "lat"], lijst1_df.loc[i, "lng"]],
                    tooltip=lijst1_df.loc[i, 'naam'],
                    radius=5,
                    fill=True,
                    color='blue',
                    fill_opacity=0.6
                ).add_to(group_uk)

        # FeatureGroup aangemaakt waarvan de vluchten niet bekend zijn.
        group_onbekend = folium.FeatureGroup(name="Geen vluchtgegevens").add_to(m)

        for i in lijst2_df.index:
            if lijst2_df.loc[i, 'land'] == 'Netherlands':
                folium.CircleMarker(
                    location=[lijst2_df.loc[i, "lat"], lijst2_df.loc[i, "lng"]],
                    tooltip=lijst2_df.loc[i, 'naam'],
                    radius=2,
                    fill=True,
                    color='black',
                    fill_opacity=0.6
                ).add_to(group_onbekend)
            elif lijst2_df.loc[i, 'land'] == 'Germany':
                folium.CircleMarker(
                    location=[lijst2_df.loc[i, "lat"], lijst2_df.loc[i, "lng"]],
                    tooltip=lijst2_df.loc[i, 'naam'],
                    radius=2,
                    fill=True,
                    color='black',
                    fill_opacity=0.6
                ).add_to(group_onbekend)
            elif lijst2_df.loc[i, 'land'] == 'United Kingdom':
                folium.CircleMarker(
                    location=[lijst2_df.loc[i, "lat"], lijst2_df.loc[i, "lng"]],
                    tooltip=lijst2_df.loc[i, 'naam'],
                    radius=2,
                    fill=True,
                    color='black',
                    fill_opacity=0.6
                ).add_to(group_onbekend)

        # Voeg een legenda toe
        folium.map.LayerControl('topright', collapsed=False).add_to(m)
        m.get_root().html.add_child(folium.Element(legend_html))

        # Toon de kaart
        st_data = st_folium(m, width = 700, height = 500)
        

elif option == 'Lijndiagram':

    # Nogmaals de gemergede dataset
    merged_df = pd.merge(df, df2, left_on = "bestemming", right_on = "ICAO", how = "inner")
    vlucht_bekend = merged_df[merged_df['land'].isin(df_zoom["land"])]
    uniek = vlucht_bekend["ICAO"].unique()
    df_uniek = vlucht_bekend.drop_duplicates(subset=['ICAO'])

    st.title("Lijndiagram")
    
    # Streamlit-app voor gdf_nederland
    st.header('Aantal vliegtuigen aan de grond per land en per vliegveld')

    # Creëer de dropdown selectie en geef het nieuwe labels
    vertaal_landen = {
        'Netherlands': 'Nederland',
        'Germany': 'Duitsland',
        'United Kingdom': 'Verenigd Koninkrijk'
    }

    vlucht_bekend['land'] = vlucht_bekend['land'].replace(vertaal_landen)

    selected_name = st.selectbox("Selecteer een land:", vlucht_bekend['land'].unique())
    if selected_name:
        selected_airports = vlucht_bekend[vlucht_bekend['land'] == selected_name]['naam'].unique()
        selected_airport = st.selectbox("Selecteer een luchthaven:", selected_airports)

        # Filter de dataset op basis van de geselecteerde waarden
        vlucht_bekend['datum'] = pd.to_datetime(vlucht_bekend['datum'])
        filtered_data = vlucht_bekend[(vlucht_bekend['land'] == selected_name) & (vlucht_bekend['naam'] == selected_airport)]
        grouped_df = filtered_data.groupby(filtered_data['datum'].dt.date).size().reset_index(name='Aantal_vliegtuigen')

        # Maak de Plotly-grafiek
        fig = go.Figure(go.Scatter(
            x=grouped_df["datum"],
            y=grouped_df['Aantal_vliegtuigen'],
            name=selected_airport, line_shape='hvh'
        ))

        # Voeg range slider toe
        fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                    label="1d",
                                    step="day",
                                    stepmode="backward"),
                            dict(count=7,
                                    label="1w",
                                    step="day",
                                    stepmode="backward"),
                            dict(count=1,
                                    label="1m",
                                    step="month",
                                    stepmode="backward"),
                            dict(count=1,
                                    label="1y",
                                    step="year",
                                    stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )       
            )

        #Update de layout van de grafiek
        fig.update_layout(
            title=f"Data voor {selected_airport} in {selected_name}",
            xaxis_title="Datum (2019 t/m 2020)",
            yaxis_title="Aantal vliegtuigen aan de grond"
        )

        # Toon de grafiek
        st.plotly_chart(fig)
        
        
        
        
else:

    # Nogmaals de gemergede dataset
    merged_df = pd.merge(df, df2, left_on = "bestemming", right_on = "ICAO", how = "inner")
    vlucht_bekend = merged_df[merged_df['land'].isin(df_zoom["land"])]
    uniek = vlucht_bekend["ICAO"].unique()
    df_uniek = vlucht_bekend.drop_duplicates(subset=['ICAO'])
    
    st.title('Voorspellingsmodel')
    st.write('Om te voorspellen of een vliegtuig vertraging oploopt of niet, is per vlucht voorspeld of deze vertraagd zal zijn of niet. Het model zelf is 67,1% accuraat. Van alle vluchten die vertraging opliepen, kon het model dit voor 77% correct voorspellen.')

    # Streamlit titel
    st.write("### Barplot:")
    
    df_model = vlucht_bekend[["datum", "tijd_gepland", "tijd", "vliegtuigtype", "bestemming", "aankomend", "naam", "stad", "land"]]
    df_model["datum"] = pd.to_datetime(df_model["datum"], format = "%d/%m/%Y")
    df_model = df_model[df_model['datum'] <= '2020-02-29']

    df_model["tijd_gepland"] = pd.to_datetime(df_model["tijd_gepland"], format='%H:%M:%S')
    df_model["tijd"] = pd.to_datetime(df_model["tijd"], format='%H:%M:%S')

    # Bereken het verschil in minuten tussen de kolommen "tijd_gepland" en "tijd"
    df_model["vertraging"] = (df_model["tijd"] - df_model["tijd_gepland"]).dt.total_seconds() // 60
    df_model["maand"] = df_model["datum"].dt.strftime('%B')

    df_model["dagdeel"] = np.where(
        (df_model["tijd_gepland"].dt.hour < 6), "nacht",
        np.where((df_model["tijd_gepland"].dt.hour < 12), "ochtend",
                np.where((df_model["tijd_gepland"].dt.hour < 18), "middag", "avond"))
    )
    
    df_model["vertraging"] = np.where(df_model["vertraging"] > 0, 1, 0)
    
    df_model = df_model[["vliegtuigtype", "bestemming", "stad", "vertraging", "maand", "dagdeel"]]

    # Zorg ervoor dat alle kolommen numeriek zijn door de LabelEncoder te gebruiken voor categorische kolommen
    label_encoders = {}
    for col in df_model.columns:
        label_encoders[col] = LabelEncoder()
        df_model[col] = label_encoders[col].fit_transform(df_model[col])
    
    # Definieer features (inputvariabelen) en target (vertraging)
    X = df_model.drop("vertraging", axis=1)
    y = df_model["vertraging"]
    
    # Split de data in trainings- en testsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creëer het Decision Tree-model
    model = DecisionTreeClassifier()
    
    # Train het model
    model.fit(X_train, y_train)
    
    # Voorspel vertragingen op de testset
    predictions = model.predict(X_test)
    
    # Bereken de nauwkeurigheid van het model
    accuracy = accuracy_score(y_test, predictions)

    # Haal de feature importance scores op
    feature_importance = model.feature_importances_
    
    # Maak een DataFrame van feature importance scores en bijbehorende kenmerken
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    
    # Sorteer de DataFrame op belangrijkheid
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Visualiseer de feature importance scores
    fig1 = go.Figure(go.Bar(
                x=feature_importance_df['Feature'],
                y=feature_importance_df['Importance']
            ))

    fig1.update_layout(
            title=('Belang van de variabelen in het model'),
            xaxis_title='Variabelen',
            yaxis_title='Score'
        )
    
    st.plotly_chart(fig1)

    st.write("### Beslisboom:")

    fig2 = plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X.columns, class_names=["Geen vertraging", "Vertraging"], filled=True, max_depth=2)
    st.pyplot(fig2)
        