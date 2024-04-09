import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import st_folium
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

modelNLP = joblib.load('modele_NLP.pkl')
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('scaler_F.pkl')
modelLR = joblib.load('modelLR_F.pkl')
df_distance = pd.read_csv('distance_airports.csv')
airports = pd.read_csv('airports.csv')
review_1 = pd.read_csv("Airline_review_transformed_1.csv")
review_2 = pd.read_csv("Airline_review_transformed_2.csv")
reviews = pd.concat([review_1, review_2])

st.set_page_config(page_title="SENDAIR", page_icon=":airplane:", layout="wide")

col1, col2, col3, col4, col5 = st.columns(5)
with col3:
    st.image('Logo.png')

col1, col2, col3 = st.columns(3)
with col2:
    st.sidebar.image('LogoSixEyes.png')
st.write('')

url_img = "https://i.ibb.co/kyK17SR/Fond.png"

style_css = '''
<style>
[data-testid="stAppViewContainer"] {
background-color:#fff;
opacity: 1;
background-image: url("https://i.ibb.co/kyK17SR/Fond.png");
background-size: contain;
}
</style>
'''
st.markdown(style_css, unsafe_allow_html=True)

# Page d'accueil
def page_accueil():
    col1, col2 = st.columns(2)
    with col1 :
        st.image('home_picture.jpg', use_column_width=True)
    with col2 :
        st.markdown("<h1 style='text-align: center;'>Welcome to Sendair's application</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>All aboard and ready for take-off!</h2>", unsafe_allow_html=True)
        st.header('')
        st.header('')
        st.header('')
        st.write("<h6 style='text-align: center;'>First, step into the future with our aircraft delay prediction model!</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: center;'>Let's work together to find the answers to these delays and improve flight performance!</h6>", unsafe_allow_html=True)
        st.header('')
        st.header('')
        st.write("<h6 style='text-align: center;'>Then, let's take a look at customer feedback!</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: center;'>Get inside their heads and find out how they would rate their flight based on their opinions!</h6>", unsafe_allow_html=True)
    st.write('')
    



# Page 1
def page_1():
    df_distance.index = df_distance.columns.tolist()

    st.markdown("<h3 style='text-align: center;'>Let's predict your delay!</h3>", unsafe_allow_html=True)

    st.write('')
    st.write('')
    st.write('')

    with st.form("filters", border = False):

        st.image('Date.png', use_column_width=True)
        col1, col2, col3 = st.columns(3)
        with col1 :
            month = st.selectbox(':blue[**What month does the flight take place?**]', ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'), index = None)
        with col2 :
            day = st.selectbox(':blue[**What day is your flight?**]', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'), index = None)
        with col3 :
            hour = st.selectbox(':blue[**What time is your flight?**]', (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), index = None)


        st.write('')
        st.write('')
        st.write('')
        st.image('Company.png', use_column_width=True)
        col1, col2 = st.columns(2)
        with col1 :
            airline = st.selectbox(':blue[**What is the airline?**]', ('Alaska Airlines Inc.', 'American Airlines Inc.', 'American Eagle Airlines Inc.', 'Atlantic Southeast Airlines', 'Delta Air Lines Inc.', 'Frontier Airlines Inc.', 'Hawaiian Airlines Inc.', 'JetBlue Airways', 'Skywest Airlines Inc.', 'Southwest Airlines Co.', 'Spirit Air Lines', 'United Air Lines Inc.', 'US Airways Inc.', 'Virgin America'), index = None)
        with col2 :
            model = st.selectbox(':blue[**What is the aircraft model?**]', ('AIRBUS A319-111', 'AIRBUS A319-114', 'AIRBUS A319-132', 'AIRBUS A320-211', 'AIRBUS A320-212', 'AIRBUS A320-214', 'AIRBUS A320-232', 'BOEING 717-200', 'BOEING 737-3H4', 'BOEING 737-824', 'BOEING 737-832', 'BOEING 737-890', 'BOEING 737-8H4', 'BOEING 737-932ER', 'BOEING 737-990ER', 'BOEING 757-232', 'BOMBARDIER CL-600-2B19', 'BOMBARDIER CL-600-2C10', 'EMBRAER ERJ 170-200 LR', 'MARZ BARRY KITFOX IV', 'MCDONNELL DOUGLAS MD-88', 'OTHER'), index = None)


        st.write('')
        st.write('')
        st.write('')
        st.image('Travel.png', use_column_width=True)
        col1, col2, col3 = st.columns(3)
        with col1 :
            origin_airport = st.selectbox(':blue[**What is the origin airport?**]', tuple(sorted(df_distance.index)), index = None)
        with col2 :
            destination_airport = st.selectbox(':blue[**What is the destination airport?**]', tuple(sorted(df_distance.columns)), index = None)
        with col3 :
            dep_delay = st.selectbox(':blue[**Was your flight delayed at departure?**]', ('No delay', 'Slight delay (5 to 20 min)', 'Long delay (> 20 min)', 'Ahead of schedule (> 15 min)', "I don't know"), index = None)


        st.write('')
        st.write('')
        st.write('')
        st.image('Weather.png', use_column_width=True)
        col1, col2 = st.columns(2)
        with col1 :
            origin_weather = st.selectbox(':blue[**What is the weather forecast on departure?**]', ('Clear sky', 'Partly clouds', 'Cloudy', 'Foggy', 'Light rain', 'Rain', 'Heavy rain', 'Freezing rain', 'Light snow', 'Snow', 'Rainstorms', 'Blizzard'), index = None)
        with col2 :
            destination_weather = st.selectbox(':blue[**What is the weather forecast on arrival?**]', ('Clear sky', 'Partly clouds', 'Cloudy', 'Foggy', 'Light rain', 'Rain', 'Heavy rain', 'Freezing rain', 'Light snow', 'Snow', 'Rainstorms', 'Blizzard'), index = None)


        
        m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: rgb(0, 140, 220);
        }
        </style>""", unsafe_allow_html=True)

        st.write('')
        st.write('')
        st.write('')
        submitted = st.form_submit_button('**Take off!**')


    if submitted :

        if origin_airport != None and destination_airport != None :
            center_lat = (airports.loc[airports['AIRPORT'] == origin_airport, 'LATITUDE'].iloc[0] + airports.loc[airports['AIRPORT'] == destination_airport, 'LATITUDE'].iloc[0])/2
            center_long = (airports.loc[airports['AIRPORT'] == origin_airport, 'LONGITUDE'].iloc[0] + airports.loc[airports['AIRPORT'] == destination_airport, 'LONGITUDE'].iloc[0])/2
            point_origin = [airports.loc[airports['AIRPORT'] == origin_airport, 'LATITUDE'].iloc[0], airports.loc[airports['AIRPORT'] == origin_airport, 'LONGITUDE'].iloc[0]]
            point_destination = [airports.loc[airports['AIRPORT'] == destination_airport, 'LATITUDE'].iloc[0], airports.loc[airports['AIRPORT'] == destination_airport, 'LONGITUDE'].iloc[0]]


        if hour == None :
            hour = 12

        if dep_delay == 'Long delay (> 15 min)' :
            dep_delay = 3
        elif dep_delay == 'Slight delay (< 15 min)' :
            dep_delay = 2
        elif dep_delay == 'No delay' :
            dep_delay = 1
        elif dep_delay == 'Ahead of schedule (> 15 min)' :
            dep_delay = 0
        else :
            dep_delay = 1.5429960352067347

        if (origin_airport == None) or (destination_airport == None):
            distance = 0
        else :
            distance = df_distance.loc[origin_airport, destination_airport]

        
        name_col = ['DEPARTURE_TIME',
    'DEPARTURE_DELAY',
    'DISTANCE',
    'ARRIVAL_DELAY',
    'April',
    'December',
    'February',
    'July',
    'June',
    'November',
    'October',
    'September',
    'Monday',
    'Saturday',
    'Thursday',
    'Alaska Airlines Inc.',
    'Atlantic Southeast Airlines',
    'Delta Air Lines Inc.',
    'Frontier Airlines Inc.',
    'JetBlue Airways',
    'Skywest Airlines Inc.',
    'Spirit Air Lines',
    'United Air Lines Inc.',
    "ORI_Chicago O'Hare International Airport",
    'ORI_Dallas/Fort Worth International Airport',
    'ORI_Denver International Airport',
    'ORI_George Bush Intercontinental Airport',
    'ORI_Hartsfield-Jackson Atlanta International Airport',
    'ORI_Minneapolis-Saint Paul International Airport',
    'ORI_Pittsburgh International Airport',
    'ORI_Salt Lake City International Airport',
    "DEST_Chicago O'Hare International Airport",
    'DEST_Hartsfield-Jackson Atlanta International Airport',
    'DEST_LaGuardia Airport (Marine Air Terminal)',
    'DEST_Salt Lake City International Airport',
    'DEST_Seattle-Tacoma International Airport',
    'AIRBUS A319-111',
    'AIRBUS A319-114',
    'AIRBUS A319-132',
    'AIRBUS A320-211',
    'AIRBUS A320-212',
    'AIRBUS A320-214',
    'AIRBUS A320-232',
    'BOEING 717-200',
    'BOEING 737-3H4',
    'BOEING 737-824',
    'BOEING 737-832',
    'BOEING 737-890',
    'BOEING 737-8H4',
    'BOEING 737-932ER',
    'BOEING 737-990ER',
    'BOEING 757-232',
    'BOMBARDIER CL-600-2B19',
    'BOMBARDIER CL-600-2C10',
    'EMBRAER ERJ 170-200 LR',
    'MARZ BARRY KITFOX IV',
    'MCDONNELL DOUGLAS MD-88',
    'ORI_Blizzard',
    'ORI_Clear sky',
    'ORI_Cloudy',
    'ORI_Freezing rain',
    'ORI_Heavy rain',
    'ORI_Light rain',
    'ORI_Light snow',
    'ORI_Partly clouds',
    'ORI_Rain',
    'ORI_Rainstorms',
    'ORI_Snow',
    'DEST_Blizzard',
    'DEST_Clear sky',
    'DEST_Foggy',
    'DEST_Freezing rain',
    'DEST_Heavy rain',
    'DEST_Light rain',
    'DEST_Light snow',
    'DEST_Rain',
    'DEST_Rainstorms',
    'DEST_Snow']
        
        list_zero = [0] * len(name_col)
        test = pd.DataFrame([list_zero], columns = name_col)

        test.loc[0,'DEPARTURE_TIME'] = hour
        test.loc[0,'DEPARTURE_DELAY'] = dep_delay
        test.loc[0,'DISTANCE'] = distance

        if month in name_col :
            test.loc[0,month] = 1
        if day in name_col :
            test.loc[0,day] = 1
        if airline in name_col :
            test.loc[0,airline] = 1
        if model in name_col :
            test.loc[0,model] = 1
        
        if origin_airport != None :
            origin_airport = 'ORI_' + origin_airport
        if destination_airport != None :
            destination_airport = 'DEST_' + destination_airport
        if origin_weather != None :
            origin_weather = 'ORI_' + origin_weather
        if destination_weather != None :
            destination_weather = 'DEST_' + destination_weather

        if origin_airport in name_col :
            test.loc[0,origin_airport] = 1
        if destination_airport in name_col :
            test.loc[0,destination_airport] = 1
        if origin_weather in name_col :
            test.loc[0,origin_weather] = 1
        if destination_weather in name_col :
            test.loc[0,destination_weather] = 1


        X_test = test.drop(columns = 'ARRIVAL_DELAY')
        X_test_scaled = scaler.transform(X_test)
        resultat = modelLR.predict(X_test_scaled)
        probability = modelLR.predict_proba(X_test_scaled)
        if resultat == 0 :
            probability = probability[0][0] + 0.75 * probability[0][1]
        elif resultat == 8:
            probability = 0.75 * probability[0][7] + probability[0][8]
        else :
            probability = 0.5 * probability[0][resultat - 1] + probability[0][resultat] + 0.5 * probability[0][resultat + 1]
        probability = int(((probability + 1) / 2) * 100)


        if resultat == 0 :
            text_resultat = '<span style="color:green;">Ahead of schedule</span>'
        if resultat == 1 :
            text_resultat = '<span style="color:green;">On time</span>'
        if resultat == 2 :
            text_resultat = '<span style="color:orange;">Estimated delay : 5 minutes</span>'
        if resultat == 3 :
            text_resultat = '<span style="color:orange;">Estimated delay : 15 minutes</span>'
        if resultat == 4 :
            text_resultat = '<span style="color:orange;">Estimated delay : 30 minutes</span>'
        if resultat == 5 :
            text_resultat = '<span style="color:orange;">Estimated delay : 45 minutes</span>'
        if resultat == 6 :
            text_resultat = '<span style="color:red;">Estimated delay : 1 hour</span>'
        if resultat == 7 :
            text_resultat = '<span style="color:red;">Estimated delay : 1 hour and a half</span>'
        if resultat == 8 :
            text_resultat = '<span style="color:red;">Estimated delay : Over 2 hours</span>'


        #col1 = st.columns(1)[0]
        with st.container(border = True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<h1 style='text-align: left;'>{text_resultat}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: left;'>Fiability : {str(probability)}%</h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: left;'>Distance : {str(distance)} miles</h3>", unsafe_allow_html=True)
            with col2:
                st.image('Enjoy_your_flight.png', use_column_width=True)


        if origin_airport != None and destination_airport != None :
            col1 = st.columns(1)[0]
            with col1:
                if distance <= 200 :
                    m = folium.Map(location=[center_lat, center_long],zoom_start=7)
                elif distance <= 800 :
                    m = folium.Map(location=[center_lat, center_long],zoom_start=6)
                elif distance <= 2500 :
                    m = folium.Map(location=[center_lat, center_long],zoom_start=5)
                else :
                    m = folium.Map(location=[center_lat, center_long],zoom_start=4)
                folium.Marker(location=point_origin,popup=origin_airport).add_to(m)
                folium.Marker(location=point_destination,popup=destination_airport).add_to(m)
                if resultat <= 1 :
                    folium.PolyLine(locations=[point_origin, point_destination], color='green').add_to(m)
                elif resultat <= 5 :
                    folium.PolyLine(locations=[point_origin, point_destination], color='orange').add_to(m)
                else :
                    folium.PolyLine(locations=[point_origin, point_destination], color='red').add_to(m)
                st_folium(m, height = 850, width = 1500, returned_objects=[])




# Page 2
def page_2():
    st.markdown("<h3 style='text-align: center;'>Be part of your users' experience!</h3>", unsafe_allow_html=True)

    st.write('')
    st.write('')
    st.write('')


    st.image('Comment.png', use_column_width=True)

    with st.container(border = True):

        txt = st.text_area(
        "",
        )

        if txt != "" :

            # Vectorisation de la chaîne de texte
            test_pos_CV = vectorizer.transform([txt])

            # Calcul du score
            score_proba = modelNLP.predict_proba(test_pos_CV)[:, 1] * 5

            # Couleur de score_proba
            if score_proba[0] <= 2:
                resultat = '<span style="color:red;">Negative feedback</span>'
                score_color = 'red'
            elif score_proba[0] <= 3:
                resultat = '<span style="color:orange;">Neutral feedback</span>'
                score_color = 'orange'
            else:
                score_color = 'green'
                resultat = '<span style="color:green;">Positive feedback</span>'

            # Afficher le résultat et le score
            st.write('')
            st.markdown("<h5 style='text-align: center;'>Customer satisfaction prediction :</h5>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>{resultat}</h3>", unsafe_allow_html=True)
            st.write('')
            st.markdown("<h5 style='text-align: center;'>Note prediction :</h5>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color:{score_color};'>{score_proba[0].round(2)}</h3>", unsafe_allow_html=True)
            if score_color == 'orange':
                st.markdown("<h5 style='text-align: center;'>Medium score, read the review for a better understanding.</h5>", unsafe_allow_html=True)

    st.write('')
    st.write('')
    st.write('')
    st.image('Reviews.png', use_column_width=True)

    df_rating = reviews[reviews['Overall_Rating'].notnull()]
    df_nan = reviews[reviews['Overall_Rating'].isnull()]
    rating_good = df_rating.loc[df_rating['Sentiment'] == 'good']
    rating_bad = df_rating.loc[df_rating['Sentiment'] == 'bad']
    best_review = rating_good.loc[(rating_good['Note_weighting'] == 5) & (rating_good['Verified'] == True)].sort_values(['Note_weighting', 'Compteur'], ascending= False)
    best_review = best_review[['Airline Name','Review_Title','Review']]
    note_avis_verified = reviews.loc[reviews['Verified'] == True]
    note_avis_not_verified = reviews.loc[reviews['Verified'] == False]
    note_avis_all = reviews

    with st.container(border = True):
        col1, col2, col3 = st.columns(3)

        #Note générale vérifié
        with col1:
            st.markdown(f"<h4 style='text-align: center;'>Note générale des avis vérifiés ({len(note_avis_verified)})</h4>", unsafe_allow_html=True)
            note_avis_verified = note_avis_verified['Note_weighting'].sum()/len(note_avis_verified)
            st.markdown(f"<h2 style='text-align: center;'>{note_avis_verified.round(2)}</h2>", unsafe_allow_html=True)

        #Note générale non vérifié
        with col2:
            st.markdown(f"<h4 style='text-align: center;'>Note générale des avis non vérifiés ({len(note_avis_not_verified)})</h4>", unsafe_allow_html=True)
            note_avis_not_verified = note_avis_not_verified['Note_weighting'].sum()/len(note_avis_not_verified)
            st.markdown(f"<h2 style='text-align: center;'>{note_avis_not_verified.round(2)}</h2>", unsafe_allow_html=True)    

        #Note générale all
        with col3:
            st.markdown(f"<h4 style='text-align: center;'>Note générale de tous les avis ({len(note_avis_all)})</h4>", unsafe_allow_html=True)
            note_avis_all = note_avis_all['Note_weighting'].sum()/len(note_avis_all)
            st.markdown(f"<h2 style='text-align: center;'>{note_avis_all.round(2)}</h2>", unsafe_allow_html=True)
    
        st.write('')
        st.write('')
        st.write('')

        #Note générale par catégorie
        st.markdown("<h4 style='text-align: center;'>Note moyenne des avis vérifiés par catégorie</h4>", unsafe_allow_html=True)  
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            #Note générale SIEGE vérifié
            st.markdown("<h5 style='text-align: center;'>Seat Comfort</h5>", unsafe_allow_html=True)
            siege = reviews.loc[(reviews['Verified'] == True) & (reviews['Seat Comfort'].notnull())]
            siege = siege['Seat Comfort'].sum()/len(siege)
            st.markdown(f"<h3 style='text-align: center;'>{siege.round(2)}</h3>", unsafe_allow_html=True)
        with col2:
        #Note générale CREW vérifié
            st.markdown("<h5 style='text-align: center;'>Cabin Staff Service</h5>", unsafe_allow_html=True)
            crew = reviews.loc[(reviews['Verified'] == True) & (reviews['Cabin Staff Service'].notnull())]
            crew = crew['Cabin Staff Service'].sum()/len(crew)
            st.markdown(f"<h3 style='text-align: center;'>{crew.round(2)}</h3>", unsafe_allow_html=True)
        with col3:
        #Note générale FOOD vérifié
            st.markdown("<h5 style='text-align: center;'>Food & Beverages</h5>", unsafe_allow_html=True)
            alimentation = reviews.loc[(reviews['Verified'] == True) & (reviews['Food & Beverages'].notnull())]
            alimentation = alimentation['Food & Beverages'].sum()/len(alimentation)
            st.markdown(f"<h3 style='text-align: center;'>{alimentation.round(2)}</h3>", unsafe_allow_html=True)
        with col4:
        #Note générale GROUND vérifié
            st.markdown("<h5 style='text-align: center;'>Ground Service</h5>", unsafe_allow_html=True)
            ground = reviews.loc[(reviews['Verified'] == True) & (reviews['Ground Service'].notnull())]
            ground = ground['Ground Service'].sum()/len(ground)
            st.markdown(f"<h3 style='text-align: center;'>{ground.round(2)}</h3>", unsafe_allow_html=True)
        with col5:
        #Note générale INFLIGHT vérifié
            st.markdown("<h5 style='text-align: center;'>Inflight Entertainment</h5>", unsafe_allow_html=True)
            service = reviews.loc[(reviews['Verified'] == True) & (reviews['Inflight Entertainment'].notnull())]
            service = service['Inflight Entertainment'].sum()/len(service)
            st.markdown(f"<h3 style='text-align: center;'>{service.round(2)}</h3>", unsafe_allow_html=True)
        with col6:
        #Note générale WIFI vérifié
            st.markdown("<h5 style='text-align: center;'>Wifi & Connectivity</h5>", unsafe_allow_html=True)
            wifi = reviews.loc[(reviews['Verified'] == True) & (reviews['Wifi & Connectivity'].notnull())]
            wifi = wifi['Wifi & Connectivity'].sum()/len(wifi)
            st.markdown(f"<h3 style='text-align: center;'>{wifi.round(2)}</h3>", unsafe_allow_html=True)
        with col7:
        #Note générale MONEY vérifié
            st.markdown("<h5 style='text-align: center;'>Value For Money</h5>", unsafe_allow_html=True)
            money = reviews.loc[(reviews['Verified'] == True) & (reviews['Value For Money'].notnull())]
            money = money['Value For Money'].sum()/len(money)
            st.markdown(f"<h3 style='text-align: center;'>{money.round(2)}</h3>", unsafe_allow_html=True)


    st.write('')
    st.write('')
    st.write('')
    
    st.markdown("<h2 style='text-align: center;'>Analyses complémentaires</h2>", unsafe_allow_html=True) 
    # Créer la figure et les axes avec Matplotlib
    fig, ax = plt.subplots(figsize = (20,8), facecolor='none')
    sns.histplot(data=reviews, x='len_review', alpha=1)
    plt.title('Histogramme de la longueur des critiques')
    ax.set_ylabel("Nombre d'avis")
    ax.set_xlabel("Nombre de caractères")
    st.pyplot(fig)

    st.write('')
    st.write('')
    st.write('')
    with st.container(border = True):
        col1, col2 = st.columns(2)
        with col1:
            # Créer la heatmap
            numeri_df = reviews.select_dtypes(include = 'number')
            heatmap_fig = px.imshow(numeri_df.corr(),
                                    labels=dict(color="Corrélation"),
                                    color_continuous_scale="RdBu_r",
                                    zmax=1,
                                    title="Heatmap de corrélation générale")
            heatmap_fig.update_layout(coloraxis_colorbar=dict(title="Corrélation", tickvals=[-1, 0, 1]), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(heatmap_fig)  
        with col2:
            fig, ax = plt.subplots(figsize = (20,8), facecolor='none')
            fig = px.box(reviews, y='len_review', title='Boxplot de la longueur du texte')
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig)

        col1, col2 = st.columns(2)
        with col1:
            # Heatmap entre longueur du texte et la note donnée
            heatmap_fig = px.imshow(reviews[['Note_weighting', 'len_review']].corr(),
                                    labels=dict(color="Correlation"),
                                    color_continuous_scale="RdBu_r",
                                    zmin=-1, zmax=1,
                                    title="Correlation entre longueur texte et note pondérée")
            heatmap_fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(heatmap_fig)
        with col2:
        # Heatmap entre la recommandation et la note donnée
            heatmap_fig = px.imshow(reviews[['Note_weighting', 'Recommended']].corr(),
                                    labels=dict(color="Correlation"),
                                    color_continuous_scale="RdBu_r",
                                    zmin=-1, zmax=1,
                                    title="Correlation entre recommandation et note pondérée")
            heatmap_fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(heatmap_fig)

    st.write('')
    st.write('')
    st.write('')
    with st.container(border = True):
        col1, col2 = st.columns(2)
        with col1:
        # Calculer le nombre d'avis ayant une Note_weighting de 5
            review_5_count = (reviews['Note_weighting'] == 5).sum()
            # Calculer le nombre total d'avis
            total_reviews_count = len(reviews)
            # Calculer le nombre d'avis ayant une Note_weighting différente de 5
            other_reviews_count = total_reviews_count - review_5_count
            # Créer un camembert
            fig, ax = plt.subplots(figsize=(20, 8), facecolor='none')
            plt.pie([review_5_count, other_reviews_count], autopct='%1.1f%%', startangle=90)
            plt.title("Proportion d'avis avec une Note_weighting de 5.0 par rapport à toutes les autres notes", fontsize=20)
            plt.legend(labels=['5/5', 'Autres'], loc='upper right', fontsize=16)
            plt.axis('equal')  # Aspect ratio égal pour s'assurer que le camembert est dessiné comme un cercle
            # Afficher le diagramme circulaire dans Streamlit
            st.pyplot(fig)
        with col2:
            # Compter les recommandations ou les non recommandations
            numeric_count = reviews['Verified'].apply(lambda x: 'Yes' if x == True else 'No').value_counts()
            # Créer un camembert
            fig, ax = plt.subplots(figsize=(20, 8), facecolor='none')
            plt.pie(numeric_count, autopct='%1.1f%%', startangle=90)
            plt.title('Avis vérifiés VS avis non vérifiés', fontsize=20)
            plt.legend(labels=numeric_count.index, loc='upper right', fontsize=16)
            plt.axis('equal')
            # Afficher le diagramme circulaire dans Streamlit
            st.pyplot(fig)
        with col1:
            # Compter les valeurs numériques et non numériques dans la colonne 'Note'
            numeric_count = reviews['Note_weighting'].apply(lambda x: 'Note' if pd.notnull(x) else 'Sans note').value_counts()
            # Créer un diagramme circulaire
            fig, ax = plt.subplots(figsize=(20, 8), facecolor='none')
            plt.pie(numeric_count, autopct='%1.1f%%', startangle=90)
            plt.title('Répartition des avis notés VS non notés', fontsize=20)
            plt.legend(labels=numeric_count.index, loc='upper right', fontsize=16)
            plt.axis('equal')
            # Afficher le diagramme circulaire dans Streamlit
            st.pyplot(fig)
        with col2:
            # Compter les recommandations ou les non recommandations
            numeric_count = reviews['Recommended'].apply(lambda x: 'Recommandé' if x == 1 else 'Non recommandé').value_counts()
            # Créer un diagramme circulaire
            fig, ax = plt.subplots(figsize=(20, 8), facecolor='none')
            plt.pie(numeric_count, autopct='%1.1f%%', startangle=90)
            plt.title('Compagnies recommandées VS Compagnies non recommandées', fontsize=20)
            plt.legend(labels=numeric_count.index, loc='upper right', fontsize=16)
            plt.axis('equal')
            # Afficher le diagramme circulaire dans Streamlit
            st.pyplot(fig)

    st.write('')
    st.write('')
    st.write('')
    with st.container(border = True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            aircraft_good = reviews.loc[reviews['Note_weighting'] == 5]
            aircraft_good_sorted = aircraft_good.groupby('Aircraft').agg({'Note_weighting' : 'count'}).sort_values(by = 'Note_weighting', ascending=False).head(5)
            # Créer un graphique à barres
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='none')
            sns.barplot(aircraft_good_sorted, y = 'Aircraft', x = 'Note_weighting')
            plt.title('Top 5 avions', fontsize=20)
            plt.ylabel('Avions', fontsize=16)
            plt.xlabel('Nombre d\'avis avec note égal à 5', fontsize=16)
            plt.tight_layout()
            ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
            # Afficher le graphique à barres dans Streamlit
            st.pyplot(fig)

        with col2:
            # Sélectionner les données pour les avis avec une Note_weighting inférieure ou égale à 1
            aircraft_bad = reviews.loc[reviews['Note_weighting'] <= 1]
            aircraft_bad_sorted = aircraft_bad.groupby('Aircraft').agg({'Note_weighting' : 'count'}).sort_values(by = 'Note_weighting', ascending=False).head(5)
            # Créer un graphique à barres
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='none')
            sns.barplot(aircraft_bad_sorted, y = 'Aircraft', x = 'Note_weighting')
            plt.title('Flop 5 avions', fontsize=20)
            plt.ylabel('Avions', fontsize=16)
            plt.xlabel('Nombre d\'avis inf. à 1', fontsize=16)
            plt.tight_layout()
            ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
            # Afficher le graphique à barres dans Streamlit
            st.pyplot(fig)

        with col3:
            # Sélectionner les données pour les avis avec une Note_weighting de 5
            company_good = reviews.loc[reviews['Note_weighting'] == 5]
            company_good_sorted = company_good.groupby('Airline Name').agg({'Note_weighting' : 'count'}).sort_values(by = 'Note_weighting', ascending=False).head(5)
            # Créer un graphique à barres
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='none')
            sns.barplot(company_good_sorted, y = 'Airline Name', x = 'Note_weighting')
            plt.title('Top 5 compagnies', fontsize=20)
            plt.ylabel('Compagnies', fontsize=16)
            plt.xlabel('Nombre d\'avis avec note égal à 5', fontsize=16)
            plt.tight_layout()
            ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
            # Afficher le graphique à barres dans Streamlit
            st.pyplot(fig)

        with col4:
            # Sélectionner les données pour les avis avec une Note_weighting inférieure ou égale à 1
            company_bad = reviews.loc[reviews['Note_weighting'] <= 1]
            company_bad_sorted = company_bad.groupby('Airline Name').agg({'Note_weighting' : 'count'}).sort_values(by = 'Note_weighting', ascending=False).head(5)
            # Créer un graphique à barres
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='none')
            sns.barplot(company_bad_sorted, y = 'Airline Name', x = 'Note_weighting')
            plt.title('Flop 5 compagnies', fontsize=20)
            plt.ylabel('Compagnies', fontsize=16)
            plt.xlabel('Nombre d\'avis inf. à 1', fontsize=16)
            plt.tight_layout()
            ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
            # Afficher le graphique à barres dans Streamlit
            st.pyplot(fig)

    st.write('')
    st.write('')
    st.write('')
    with st.container(border = True):
        col1, col2 = st.columns(2)
        with col1:
            top_review = rating_good.loc[(rating_good['Note_weighting'] == 5) & (rating_good['Sum_note'] == 35) & (rating_good['Verified'] == True)]
            top_review[['Airline Name', 'Review_Title', 'Review', 'Aircraft', 'Date Flown', 'Note_weighting', 'Sum_note', 'Verified']]
            st.markdown(f"<h4 style='text-align: center;'>Number of best feedback : {len(top_review)}</h4>", unsafe_allow_html=True)
        with col2:
            flop_review = rating_bad.loc[(rating_bad['Note_weighting'] == 0)]
            flop_review[['Airline Name', 'Review_Title', 'Review', 'Aircraft', 'Date Flown', 'Note_weighting', 'Sum_note', 'Verified']]
            st.markdown(f"<h4 style='text-align: center;'>Number of worst feedback : {len(flop_review)}</h4>", unsafe_allow_html=True)


    st.write('')
    st.write('')
    st.write('')
    st.image('Increase.png', use_column_width=True)
    nombre_avis = st.slider("Nombre d'avis à ajouter", min_value=0, max_value=100000, step=1)
    augmentation = (reviews['Note_weighting'].sum() + 5*nombre_avis) / (len(reviews) + nombre_avis)
    st.markdown(f"<h3 style='text-align: center;'>Nouveau taux grâce à l'ajout de {nombre_avis} avis 5/5 : {augmentation.round(2)}</h3>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='text-align: center;'>Pourcentage des avis positif 5/5 ajouté pour faire progresser la note générale : {round((nombre_avis/(len(reviews)+nombre_avis) *100), 2)}%</h3>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='text-align: center;'>Pourcentage des avis positif 5/5 ajouté dans les avis 5/5 existant : {round((nombre_avis/(len(reviews.loc[(reviews['Note_weighting'] == 5.0) & (reviews['Verified'] == 1)])) *100),2)}%</h3>", unsafe_allow_html=True)


# Affichage des pages
def main():
    st.sidebar.markdown("<h1 style='text-align: center;'>Main menu</h1>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Home", "Delay prediction", "User experience"])

    if page == "Home":
        page_accueil()
    elif page == "Delay prediction":
        page_1()
    elif page == "User experience":
        page_2()

if __name__ == "__main__":
    main()
