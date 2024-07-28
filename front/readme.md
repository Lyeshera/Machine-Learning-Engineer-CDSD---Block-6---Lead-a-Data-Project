Démarche de création de l'app:

cd dans le répertoire du front
heroku login
heroku container:login
heroku create forex-api-jedha-front
heroku container:push web -a forex-api-jedha-front
heroku container:release web -a forex-api-jedha-front
heroku open -a forex-api-jedha-front

app dispo ici:  https://forex-api-jedha-front-5076d7fe9f64.herokuapp.com/dashboard