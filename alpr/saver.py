"""
Logica relacionada con guardar detecciones en una BD.
"""
import sqlite3
from pathlib import Path


class SqlSaver:
    """
    Se encarga de guarda la información en una
    base de datos local (SQLite)
    """

    def __init__(self, frequency_insert: int = 10, db_path: str = "db/plates.db"):
        """
        frequency_insert:   que tan seguido cantidad de patentes/len(unique_plates)
                            hacer un insert a la base de datos
        """
        # self.batch_count = 0
        self.unique_plates = set()
        self.frequency_insert = frequency_insert
        # Creo si no existe la capeta/s
        db_path = Path(db_path)
        Path(db_path.parent).mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        cursor = self.conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS plates
                            (patente text)"""
        )
        self.conn.commit()
        cursor.close()

    def update_in_memory(self, plates: list) -> None:
        """
        Actualiza en el set que vive en memoria
        """
        self.unique_plates.update(plates)
        # Trigger insert si > frequency_insert
        if len(self.unique_plates) > self.frequency_insert:
            # Inserto en BD
            self.insert_in_disk()
            # Borrar el set
            self.unique_plates.clear()

    def insert_in_disk(self):
        """
        Inserta en SQLite
        """
        cursor = self.conn.cursor()
        cursor.executemany(
            "insert into plates(patente) values (?)",
            [(plate,) for plate in self.unique_plates],
        )
        self.conn.commit()
        cursor.close()

    def __del__(self):
        # Commit cualquier cambio no guardado
        self.conn.commit()
        self.conn.close()
