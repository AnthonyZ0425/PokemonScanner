import sqlite3
from ..key import API_KEY, DB_PATH
from pokemontcgsdk import RestClient, Card

RestClient.configure(api_key=API_KEY)

connect = sqlite3.connect(DB_PATH)
cursor = connect.cursor()

# Make a GET request to fetch data from the API
def storePokemonData():
  page = 1
  total_cards = 0

  # Fetch Pokemon cards page by page (pagination) from TCG API
  while True:
    cards = Card.where(page=page, pageSize=250)
    if not cards:
      break

    # Use batch insert
    bulk_data = [
      (card.name, card.set.series, card.set.name, card.images.small)
      for card in cards
    ]

    cursor.executemany('''
    INSERT INTO pokemonList (name, series, setName, imageURL)
    VALUES (?, ?, ?, ?)
    ''', bulk_data)
    
    total_cards += len(cards)
    page += 1
  # Commit changes to the database
  connect.commit()
  print(f"Total cards inserted: {total_cards}")

try:
  storePokemonData()
except Exception as e:
  print(f"An error occurred: {e}")
finally:
  connect.close()