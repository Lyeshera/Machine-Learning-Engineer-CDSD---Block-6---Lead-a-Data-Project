Démarche de création de l'app:

cd dans le répertoire du back
heroku login
heroku container:login
heroku create forex-api-jedha-back
heroku container:push web -a forex-api-jedha-back
heroku container:release web -a forex-api-jedha-back
heroku open -a forex-api-jedha-back

app dispo ici: https://forex-api-jedha-back-14619c7a835e.herokuapp.com/docs