import sqlite3
from ..key import API_KEY
from pokemontcgsdk import RestClient, Card

RestClient.configure(api_key="")

connect = sqlite3.connect("pokemon_card.db")
cursor = connect.cursor()

# Make a GET request to fetch data from the API
def storePokemonData():
   
  # Fetch all Pokemon cards from TCG API
  cards = Card.all()

  for card in cards:
    name = card.name
    series = card.set.series
    setName = card.set.name
    imageURL = card.images.small
    
    cursor.execute('''
    INSERT INTO pokemonList (name, series, setName, imageURL)
    VALUES (?, ?, ?, ?)
    ''', (name, series, setName, imageURL))

  # Commit changes to the database
  connect.commit()

  print(f"{len(cards)} cards inserted into the database.")

storePokemonData()

connect.close()