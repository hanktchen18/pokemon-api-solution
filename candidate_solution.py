# candidate_solution.py
import sqlite3
import os
from fastapi import FastAPI, HTTPException
from typing import List, Optional
import uvicorn
import requests
import random
import time
from functools import lru_cache

# --- Constants ---
DB_NAME = "pokemon_assessment.db"


# --- Database Connection ---
def connect_db() -> Optional[sqlite3.Connection]:
    """
    Task 1: Connect to the SQLite database.
    Implement the connection logic and return the connection object.
    Return None if connection fails.
    """
    if not os.path.exists(DB_NAME):
        print(f"Error: Database file '{DB_NAME}' not found.")
        return None

    connection = None
    try:
        # --- Implement Here ---

        # Install core dependencies to run the code
        # pip install fastapi
        # pip install uvicorn[standard]
        # pip install requests

        # Set timeout to 30 seconds, useful in multi-process scenarios to reduce "database is locked" errors.
        connection = sqlite3.connect(DB_NAME, timeout=30.0)

        # Enable foreign key constraints referential integrity (e.g., preventing orphaned rows when deleting)
        connection.execute("PRAGMA foreign_keys = ON")

        # WAL mode enables concurrent reads without blocking writers
        # Improves performance for multi-reader environments.
        # Instead of locking the whole database, writes go into a separate WAL file until committed.
        connection.execute("PRAGMA journal_mode = WAL")

        # NORMAL synchronous mode balances safety and performance
        # Acceptable for this use case as data is relatively static
        # FULL is slow, OFF is fastest but risky. Using NORMAL for balance considering the data is relatively static.
        connection.execute("PRAGMA synchronous = NORMAL")

        # Increase cache size to 10MB for better query performance (reduces disk I/O by keeping more frequently accessed data in memory)
        # Allocates a larger in-memory page cache (~10MB if default page size = 1KB).
        connection.execute("PRAGMA cache_size = 10000")

        # Store temporary data in memory for faster operations
        # Temporary tables, indices or sorting results are stored in RAM instead of disk.
        # This is faster for small queries but consumes more memory.
        connection.execute("PRAGMA temp_store = MEMORY")

        # Maps up to 256MB of the database file into memory for faster access
        # Allows OS to optimize read/writes without going through SQLite's buffer. (use OS page cache instead)
        # This can significantly improve performance for large databases.
        connection.execute("PRAGMA mmap_size = 268435456")  # 256MB memory mapping

        # Make querry results behave like a dictionary.
        # This allows accessing columns by name instead of index, which
        # is particularly useful for many columns.
        # Using dict-like row access for cleaner and more maintainable code.
        connection.row_factory = sqlite3.Row

        # --- End Implementation ---
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

    return connection


# --- Data Cleaning ---
def clean_database(conn: sqlite3.Connection):
    """
    Task 2: Clean up the database using the provided connection object.
    Implement logic to:
    - Remove duplicate entries in tables (pokemon, types, abilities, trainers).
      Choose a consistent strategy (e.g., keep the first encountered/lowest ID).
    - Correct known misspellings (e.g., 'Pikuchu' -> 'Pikachu', 'gras' -> 'Grass', etc.).
    - Standardize casing (e.g., 'fire' -> 'Fire' or all lowercase for names/types/abilities).
    """
    if not conn:
        print("Error: Invalid database connection provided for cleaning.")
        return

    cursor = conn.cursor()
    print("Starting database cleaning...")

    try:
        # --- Implement Here ---

        # Install core dependencies to run the code
        # pip install fastapi
        # pip install uvicorn[standard]
        # pip install requests

        # Using indexes for optimization in this project because:
        # - This project is mostly read-heavy, so indexes will help speed up queries.
        # - Indexing overhead is negligible because the database is relatively small.
        # - The performance benefits far outweigh the minimal write overhead. (O(log n) lookups instead of O(n) table scans)
        # - Better for concurrent access.


        indexes = [
            # Case-insensitive name indexes for all main tables
            # This is critiacal for API that search by name, reducing lookup time from O(n) to O(log n).
            "CREATE INDEX IF NOT EXISTS idx_pokemon_name_lower ON pokemon(LOWER(name))",
            "CREATE INDEX IF NOT EXISTS idx_types_name_lower ON types(LOWER(name))",
            "CREATE INDEX IF NOT EXISTS idx_abilities_name_lower ON abilities(LOWER(name))",
            "CREATE INDEX IF NOT EXISTS idx_trainers_name_lower ON trainers(LOWER(name))",

            # Composite index for Pokemon type lookups
            # This is the optimization for the API /pokemon/type/{type_name} endpoint that lookups both type1 and type2.
            "CREATE INDEX IF NOT EXISTS idx_pokemon_types ON pokemon(type1_id, type2_id)",

            # Foreign key indexes for join operations
            # Indexing foreign keys is helpful in a join-heavy table.
            "CREATE INDEX IF NOT EXISTS idx_tpa_pokemon ON trainer_pokemon_abilities(pokemon_id)",
            "CREATE INDEX IF NOT EXISTS idx_tpa_trainer ON trainer_pokemon_abilities(trainer_id)",
            "CREATE INDEX IF NOT EXISTS idx_tpa_ability ON trainer_pokemon_abilities(ability_id)"
        ]

        # Create indexes while handling the existing one.
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error:
                pass  # Index might already exist

        # Fix type misspellings and casing in a single optimized query
        # Batch updats with CASE instead of loops, significantly improves performance comparing to multiple single updates.
        # This enables fewer roundtrips to the database and less transaction overhead.
        cursor.execute("""
            UPDATE types SET name = CASE
                WHEN LOWER(name) = 'fire' THEN 'Fire'
                WHEN LOWER(name) = 'water' THEN 'Water'
                WHEN LOWER(name) = 'gras' THEN 'Grass'
                WHEN LOWER(name) = 'poision' THEN 'Poison'
                ELSE name
            END
            WHERE LOWER(name) IN ('fire', 'water', 'gras', 'poision')
        """)

        # Fix ability casing using batch update.
        cursor.execute("""
            UPDATE abilities SET name = CASE
                WHEN LOWER(name) = 'static' THEN 'Static'
                WHEN LOWER(name) = 'overgrow' THEN 'Overgrow'
                ELSE name
            END
            WHERE LOWER(name) IN ('static', 'overgrow')
        """)

        # Fix trainer casing using batch update.
        cursor.execute("""
            UPDATE trainers SET name = CASE
                WHEN LOWER(name) = 'misty' THEN 'Misty'
                ELSE name
            END
            WHERE LOWER(name) = 'misty'
        """)

        # Fix pokemon casing using batch update.
        cursor.execute("""
            UPDATE pokemon SET name = CASE
                WHEN LOWER(name) = 'pikuchu' THEN 'Pikachu'
                WHEN LOWER(name) = 'charmanderr' THEN 'Charmander'
                WHEN LOWER(name) = 'bulbasuar' THEN 'Bulbasaur'
                WHEN LOWER(name) = 'rattata' THEN 'Rattata'
                ELSE name
            END
            WHERE LOWER(name) IN ('pikuchu', 'charmanderr', 'bulbasuar', 'rattata')
        """)

        # Remove redundant/placeholder entries
        cursor.execute("DELETE FROM types WHERE name IN ('---', '???', '')")
        cursor.execute("DELETE FROM abilities WHERE name = 'Remove this ability'")

        # remove duplicates with proper foreign key handling
        # update foreign key references first before deleting duplicates:
        # For each foreign key column that could point at a duplicate, replace it with the ID
        # of the canonical row(the row with the smallest id among all rows sharing the same LOWER(name))
        ## type1
        cursor.execute("""
            UPDATE pokemon
            SET type1_id = (
                SELECT MIN(t2.id) FROM types t2
                WHERE LOWER(t2.name) = (
                    SELECT LOWER(t3.name) FROM types t3 WHERE t3.id = pokemon.type1_id
                )
            )
            WHERE type1_id IN (
                SELECT id FROM types
                WHERE id NOT IN (
                    SELECT MIN(id) FROM types
                    GROUP BY LOWER(name)
                )
            )
        """)
        ## type2
        cursor.execute("""
            UPDATE pokemon
            SET type2_id = (
                SELECT MIN(t2.id) FROM types t2
                WHERE LOWER(t2.name) = (
                    SELECT LOWER(t3.name) FROM types t3 WHERE t3.id = pokemon.type2_id
                )
            )
            WHERE type2_id IN (
                SELECT id FROM types
                WHERE id NOT IN (
                    SELECT MIN(id) FROM types
                    GROUP BY LOWER(name)
                )
            )
        """)

        ## abilities
        cursor.execute("""
            UPDATE trainer_pokemon_abilities
            SET pokemon_id = (
                SELECT MIN(p2.id) FROM pokemon p2
                WHERE LOWER(p2.name) = (
                    SELECT LOWER(p3.name) FROM pokemon p3 WHERE p3.id = trainer_pokemon_abilities.pokemon_id
                )
            )
            WHERE pokemon_id IN (
                SELECT id FROM pokemon
                WHERE id NOT IN (
                    SELECT MIN(id) FROM pokemon
                    GROUP BY LOWER(name)
                )
            )
        """)
        ## trainer
        cursor.execute("""
            UPDATE trainer_pokemon_abilities
            SET trainer_id = (
                SELECT MIN(t2.id) FROM trainers t2
                WHERE LOWER(t2.name) = (
                    SELECT LOWER(t3.name) FROM trainers t3 WHERE t3.id = trainer_pokemon_abilities.trainer_id
                )
            )
            WHERE trainer_id IN (
                SELECT id FROM trainers
                WHERE id NOT IN (
                    SELECT MIN(id) FROM trainers
                    GROUP BY LOWER(name)
                )
            )
        """)

        # Fix ability foreign key using batch update.
        cursor.execute("""
            UPDATE trainer_pokemon_abilities
            SET ability_id = (
                SELECT MIN(a2.id) FROM abilities a2
                WHERE LOWER(a2.name) = (
                    SELECT LOWER(a3.name) FROM abilities a3 WHERE a3.id = trainer_pokemon_abilities.ability_id
                )
            )
            WHERE ability_id IN (
                SELECT id FROM abilities
                WHERE id NOT IN (
                    SELECT MIN(id) FROM abilities
                    GROUP BY LOWER(name)
                )
            )
        """)

        # safely delete duplicates
        # After all references are repointed to canonical IDs
        # Delete every duplicate row in each table, keeping only the MIN(id) per LOWER(name)
        cursor.execute("""
            DELETE FROM types
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM types
                GROUP BY LOWER(name)
            )
        """)

        cursor.execute("""
            DELETE FROM abilities
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM abilities
                GROUP BY LOWER(name)
            )
        """)

        cursor.execute("""
            DELETE FROM trainers
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM trainers
                GROUP BY LOWER(name)
            )
        """)

        cursor.execute("""
            DELETE FROM pokemon
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM pokemon
                GROUP BY LOWER(name)
            )
        """)

        # --- End Implementation ---
        conn.commit()
        print("Database cleaning finished and changes committed.")

    except sqlite3.Error as e:
        print(f"An error occurred during database cleaning: {e}")
        conn.rollback()  # Roll back changes on error

# --- FastAPI Application ---
def create_fastapi_app() -> FastAPI:
    """
    FastAPI application instance.
    Define the FastAPI app and include all the required endpoints below.
    """
    print("Creating FastAPI app and defining endpoints...")
    app = FastAPI(title="Pokemon Assessment API")

    # --- Define Endpoints Here ---
    @app.get("/")
    def read_root():
        """
        Task 3: Basic root response message
        Return a simple JSON response object that contains a `message` key with any corresponding value.
        """
        # --- Implement here ---
        return {"message": "Welcome to Pokemon Assessment API by Ying-Han Chen!"}
        # --- End Implementation ---

    @app.get("/pokemon/ability/{ability_name}", response_model=List[str])
    def get_pokemon_by_ability(ability_name: str):
        """
        Task 4: Retrieve all Pokémon names with a specific ability.
        Query the cleaned database. Handle cases where the ability doesn't exist.
        """
        # --- Implement here ---
        # Install core dependencies to run the code
        # pip install fastapi
        # pip install uvicorn[standard]
        # pip install requests
        conn = connect_db()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        try:
            cursor = conn.cursor()

            # query using indexes for optimization
            # link pokemon to the junction table, and link junction table to abilities
            # sort results by pokemon name
            cursor.execute("""
                SELECT DISTINCT p.name
                FROM pokemon p
                INNER JOIN trainer_pokemon_abilities tpa ON p.id = tpa.pokemon_id
                INNER JOIN abilities a ON tpa.ability_id = a.id
                WHERE LOWER(a.name) = LOWER(?)
                ORDER BY p.name
            """, (ability_name,))

            results = [row[0] for row in cursor.fetchall()]

            # if not results:
            #     raise HTTPException(status_code=404, detail=f"No Pokemon found with ability '{ability_name}'")

            # Return empty list instead of 404 for better API usability
            # API Design Consistency: All endpoints return List[str] type
            # Better User Experience: Empty list indicates "query successful, but no results"
            # RESTful Principles: 404 is typically for "resource not found", not "empty query results"
            # Frontend Friendly: Frontend can uniformly handle list results, whether empty or not

            return results
        finally:
            conn.close()
        # --- End Implementation ---

    @app.get("/pokemon/type/{type_name}", response_model=List[str])
    def get_pokemon_by_type(type_name: str):
        """
        Task 5: Retrieve all Pokémon names of a specific type (considers type1 and type2).
        Query the cleaned database. Handle cases where the type doesn't exist.
        """
        # --- Implement here ---
        # Install core dependencies to run the code
        # pip install fastapi
        # pip install uvicorn[standard]
        # pip install requests
        conn = connect_db()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        try:
            cursor = conn.cursor()

            # LEFT JOIN keeps rows when type2 is NULL, which is the case for Pokemon with only one type.
            # DISTINCT prevents duplicate results.
            # ORDER BY p.name sorts results by pokemon name.
            # If no index is used, the query will be O(n*m) table scan across multiple tables.
            # With indexes, the query will be O(log n + log m) indexed lookups.
            cursor.execute("""
                SELECT DISTINCT p.name
                FROM pokemon p
                INNER JOIN types t1 ON p.type1_id = t1.id
                LEFT JOIN types t2 ON p.type2_id = t2.id
                WHERE LOWER(t1.name) = LOWER(?) OR LOWER(t2.name) = LOWER(?)
                ORDER BY p.name
            """, (type_name, type_name))

            results = [row[0] for row in cursor.fetchall()]

            # if not results:
            #     raise HTTPException(status_code=404, detail=f"No Pokemon found with type '{type_name}'")

            # Return empty list instead of 404 for better API usability
            # API Design Consistency: All endpoints return List[str] type
            # Better User Experience: Empty list indicates "query successful, but no results"
            # RESTful Principles: 404 is typically for "resource not found", not "empty query results"
            # Frontend Friendly: Frontend can uniformly handle list results, whether empty or not

            return results
        finally:
            conn.close()
        # --- End Implementation ---

    @app.get("/trainers/pokemon/{pokemon_name}", response_model=List[str])
    def get_trainers_by_pokemon(pokemon_name: str):
        """
        Task 6: Retrieve all trainer names who have a specific Pokémon.
        Query the cleaned database. Handle cases where the Pokémon doesn't exist or has no trainer.
        """
        # --- Implement here ---
        # Install core dependencies to run the code
        # pip install fastapi
        # pip install uvicorn[standard]
        # pip install requests
        conn = connect_db()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        try:
            cursor = conn.cursor()

            # query using indexes for optimization
            # sort results by pokemon name
            cursor.execute("""
                SELECT DISTINCT t.name
                FROM trainers t
                INNER JOIN trainer_pokemon_abilities tpa ON t.id = tpa.trainer_id
                INNER JOIN pokemon p ON tpa.pokemon_id = p.id
                WHERE LOWER(p.name) = LOWER(?)
                ORDER BY t.name
            """, (pokemon_name,))

            results = [row[0] for row in cursor.fetchall()]

            # if not results:
            #     raise HTTPException(status_code=404, detail=f"No trainers found with Pokemon '{pokemon_name}'")

            # Return empty list instead of 404 for better API usability
            # API Design Consistency: All endpoints return List[str] type
            # Better User Experience: Empty list indicates "query successful, but no results"
            # RESTful Principles: 404 is typically for "resource not found", not "empty query results"
            # Frontend Friendly: Frontend can uniformly handle list results, whether empty or not

            return results
        finally:
            conn.close()
        # --- End Implementation ---

    @app.get("/abilities/pokemon/{pokemon_name}", response_model=List[str])
    def get_abilities_by_pokemon(pokemon_name: str):
        """
        Task 7: Retrieve all ability names of a specific Pokémon.
        Query the cleaned database. Handle cases where the Pokémon doesn't exist.
        """
        # --- Implement here ---
        # Install core dependencies to run the code
        # pip install fastapi
        # pip install uvicorn[standard]
        # pip install requests
        conn = connect_db()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        try:
            cursor = conn.cursor()

            # query using indexes for optimization
            # sort results by pokemon name
            cursor.execute("""
                SELECT DISTINCT a.name
                FROM abilities a
                INNER JOIN trainer_pokemon_abilities tpa ON a.id = tpa.ability_id
                INNER JOIN pokemon p ON tpa.pokemon_id = p.id
                WHERE LOWER(p.name) = LOWER(?)
                ORDER BY a.name
            """, (pokemon_name,))

            results = [row[0] for row in cursor.fetchall()]

            # if not results:
            #    raise HTTPException(status_code=404, detail=f"No abilities found for Pokemon '{pokemon_name}'")

            # Return empty list instead of 404 for better API usability
            # API Design Consistency: All endpoints return List[str] type
            # Better User Experience: Empty list indicates "query successful, but no results"
            # RESTful Principles: 404 is typically for "resource not found", not "empty query results"
            # Frontend Friendly: Frontend can uniformly handle list results, whether empty or not

            return results
        finally:
            conn.close()
        # --- End Implementation ---

    # --- Implement Task 8 here ---
    # Install core dependencies to run the code
    # pip install fastapi
    # pip install uvicorn[standard]
    # pip install requests

    # Keeps up to 128 Pokemon data in cache to reduce redundant API calls.
    @lru_cache(maxsize=128)
    def _fetch_pokemon_data(pokemon_name: str):
        """Cache Pokemon data from PokeAPI to reduce external API calls."""
        try:
            pokemon_name = pokemon_name.strip().lower()
            response = requests.get(
                f"https://pokeapi.co/api/v2/pokemon/{pokemon_name}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            resp = e.response
            if resp is not None and resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Not found in PokeAPI")
            raise HTTPException(status_code=502, detail=f"PokeAPI HTTP error ({resp.status_code if resp else 'unknown'})")

        except requests.Timeout:
            raise HTTPException(status_code=504, detail="PokeAPI timeout")

        except requests.ConnectionError:
            raise HTTPException(status_code=502, detail="PokeAPI connection error")

        except requests.TooManyRedirects:
            raise HTTPException(status_code=502, detail="Too many redirects from PokeAPI")

        except requests.RequestException as e:
            # Catch-all for anything else (e.g., invalid URL, chunked encoding errors, etc.)
            raise HTTPException(status_code=502, detail=f"PokeAPI request error: {e}")

    @app.post("/pokemon/create/{pokemon_name}")
    def create_pokemon_record(pokemon_name: str):
        """
        Task 8: Create a new Pokemon record via Pokemon API.
        Fetches Pokemon data from PokeAPI and creates a record in trainer_pokemon_abilities table.
        """
        try:
            # Fetch Pokemon data from PokeAPI with caching.
            pokemon_data = _fetch_pokemon_data(pokemon_name)

            # Defensive programming, safe extraction and error handling:
            # using .get() so mssing keys don't crash the request.
            types = [type_info['type']['name'].title() for type_info in pokemon_data.get('types', [])]
            abilities = [ability_info['ability']['name'].title() for ability_info in pokemon_data.get('abilities', [])]

            conn = connect_db()
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")

            try:
                cursor = conn.cursor()

                # PERFORMANCE OPTIMIZATION: Use indexed lookup for Pokemon existence check
                # Leverages idx_pokemon_name_lower index for O(log n) performance
                # Demonstrates strategic use of database indexes for optimal query performance
                cursor.execute("SELECT id FROM pokemon WHERE LOWER(name) = LOWER(?)", (pokemon_name,))
                pokemon_result = cursor.fetchone()

                if pokemon_result:
                    pokemon_id = pokemon_result[0]
                else:
                    # If not found, resolve type1_id and type2_id by looking up/inserting the type names,
                    # and then insert the Pokemon record.

                    # UPSERT pattern for Pokemon and types
                    # Reduces database round trips by checking existence before insertion
                    type1_id = None
                    type2_id = None

                    if types:
                        # Indexed lookup for type existence check
                        cursor.execute("SELECT id FROM types WHERE LOWER(name) = LOWER(?)", (types[0],))
                        type1_result = cursor.fetchone()
                        if type1_result:
                            type1_id = type1_result[0]
                        else:
                            cursor.execute("INSERT INTO types (name) VALUES (?)", (types[0],))
                            type1_id = cursor.lastrowid

                    if len(types) > 1:
                        # dual-type Pokemon handling
                        cursor.execute("SELECT id FROM types WHERE LOWER(name) = LOWER(?)", (types[1],))
                        type2_result = cursor.fetchone()
                        if type2_result:
                            type2_id = type2_result[0]
                        else:
                            cursor.execute("INSERT INTO types (name) VALUES (?)", (types[1],))
                            type2_id = cursor.lastrowid

                    cursor.execute("INSERT INTO pokemon (name, type1_id, type2_id) VALUES (?, ?, ?)",
                                 (pokemon_name, type1_id, type2_id))
                    pokemon_id = cursor.lastrowid

                # random trainer selection
                # If no trainer found it will fail with 500 error.
                cursor.execute("SELECT id FROM trainers ORDER BY RANDOM() LIMIT 1")
                trainer_result = cursor.fetchone()
                if not trainer_result:
                    raise HTTPException(status_code=500, detail="No trainers found in database")
                trainer_id = trainer_result[0]

                # efficient batch processing and transaction management for abilities and relationships
                # Processes all abilities in a single transaction for better performance
                # Statement reuse: the same parameterized SQL gets executed repeatedly, which SQLite can cache
                # Hot cache:repeated access touches the same tables/pages, lowering I/O
                record_ids = []
                for ability_name in abilities:
                    # Use indexed lookup for ability existence check
                    cursor.execute("SELECT id FROM abilities WHERE LOWER(name) = LOWER(?)", (ability_name,))
                    ability_result = cursor.fetchone()
                    if ability_result:
                        ability_id = ability_result[0]
                    else:
                        cursor.execute("INSERT INTO abilities (name) VALUES (?)", (ability_name,))
                        ability_id = cursor.lastrowid

                    # Creates the relationship row in the junction table:
                    cursor.execute("INSERT INTO trainer_pokemon_abilities (pokemon_id, trainer_id, ability_id) VALUES (?, ?, ?)",
                                 (pokemon_id, trainer_id, ability_id))
                    record_ids.append(cursor.lastrowid)

                # Single transaction commit for atomicity -> fewer commit, saving latency.
                conn.commit()

                return {
                    "id": record_ids[0] if record_ids else pokemon_id,  # Primary record ID
                    "message": f"Successfully created Pokemon record for {pokemon_name}",
                    "record_ids": record_ids,
                    "pokemon_id": pokemon_id,
                    "trainer_id": trainer_id,
                    "abilities": abilities,
                    "types": types
                }

            finally:
                conn.close()

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    # --- End Implementation ---

    print("FastAPI app created successfully.")
    return app


# --- Main execution / Uvicorn setup (Optional - for candidate to run locally) ---
if __name__ == "__main__":
    # Ensure data is cleaned before running the app for testing
    temp_conn = connect_db()
    if temp_conn:
        clean_database(temp_conn)
        temp_conn.close()

    app_instance = create_fastapi_app()
    uvicorn.run(app_instance, host="127.0.0.1", port=8000)
