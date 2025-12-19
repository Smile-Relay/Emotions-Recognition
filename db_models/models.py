from datetime import datetime

from peewee import SqliteDatabase, Model, CharField, TextField, Check, DateTimeField
import re

db = SqliteDatabase('bottles.db')


class BaseModel(Model):
    class Meta:
        database = db


class Bottle(BaseModel):
    id = CharField(
        max_length=2,
        primary_key=True,
        constraints=[
            Check('LENGTH(id) = 2'),
            Check("UPPER(id) GLOB '[0-9A-F][0-9A-F]'")
        ]
    )

    emotion = CharField(max_length=50, null=False)

    feeling = CharField(max_length=50, null=False)

    passage = TextField(null=False)

    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'bottles'

    @classmethod
    def validate_hex_id(cls, hex_id):
        if not isinstance(hex_id, str):
            return False
        hex_id = hex_id.upper()
        pattern = re.compile(r'^[0-9A-F]{2}$')
        return bool(pattern.match(hex_id))

    def save(self, *args, **kwargs):
        if not self.validate_hex_id(self.id):
            raise ValueError(f"无效的16进制主键: {self.id}")
        self.id = self.id.upper()
        if self.created_at is None:
            self.created_at = datetime.now()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.id}: {self.emotion} - {self.feeling}"

db.connect()
db.create_tables([Bottle], safe=True)