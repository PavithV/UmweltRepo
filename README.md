SenseBox Echtzeit-Dashboard mit ML-Modul
Installation & Start

1.Klone das Repository:

git clone <repository-url>
cd <projekt-ordner>

Passe ggf. docker-compose.yml an (Sensor-ID etc.).

2.Baue und starte alle Container:

docker compose up --build -d

Prüfe, ob alle Container laufen:

docker compose ps

Öffne im Browser: http://localhost:8050
