# Väestömetsä
Väestön ennustaminen RandomForest-algoritmilla.

Sovellus löytyy osoitteesta http://vaestometsa.herokuapp.com/.

Ohjelmalla voi ennustaa väestön haluamalleensa kaupungille itse koostetulla satunnaismetsällä.

Datana käytetään Tilastokeskuksen avoimen REST API rajapinnan kautta saatavaa väestödataa kunnittain 1-vuotisiän mukaan. 

Ohjelmassa voi myös simuloida koostamaansa ennustetta ja testata saamansa ennustedataa toteutuneeseen.

Kuvaajissa esitetään koko väestön ennuste, mutta myös yksityiskohtaisen, 1-vuotisiän sisältävän Excel-tiedoston saa ladattua sivulta.

Visualisoinnit on tuotettu Dash- ja plotly -kirjastoilla. Koneoppimisessa ja datan organisoimisessa on käytössä scikit-learn ja pandas.

Katso täältä ohjeet, Dash-aplikaatioiden julkaisemiseen Herokussa:
https://dash.plotly.com/deployment

# Sovellus voitti Datamenestyjät 2021-kilpailun.
Katso alkuperäinen uutinen täältä:
https://www.stat.fi/uutinen/vaestometsa-palvelu-voitti-ensimmaisen-datamenestyjat-kilpailun
