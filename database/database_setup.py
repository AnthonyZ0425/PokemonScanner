import sqlite3

connect = sqlite3.connect('pokemon_card.db')

cursor = connect.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS pokemonList (
    id INTEGER PRIMARY KEY,
    name varchar(40) not null,
    series varchar(60) not null,
    setName varchar(60) not null,
    imageURL varchar(200) not null)
''')

connect.commit()
connect.close()