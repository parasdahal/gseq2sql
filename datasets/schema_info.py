import json

class SchemaInfo:
  def __init__(self, filepath):
    self.dbs = {}

    self.load_file(filepath)

  def load_file(self, filepath):
    with open(filepath) as file:
      data = json.load(file)
      for db in data:
        db_id = db['db_id']
        self.dbs[db_id] = {}
        for i, table_name in enumerate(db['table_names_original']):
          columns = [col[1] for col in db['column_names_original'] if col[0] == i]
          self.dbs[db_id][table_name] = columns

  def get_schema(self, db_id):
    return self.dbs[db_id]

  def get_schema_string(self, db_id):
    schema = self.get_schema(db_id)
    tokens = []
    for table in schema:
      tokens = tokens + ['[T]', table]
      for column in schema[table]:
        tokens = tokens + ['[C]', column]
    return ' '.join(tokens)

  def get_tokens(self):
    tokens = []
    for db in self.dbs:
      for table in self.dbs[db]:
        tokens.append(table)
        tokens = tokens + self.dbs[db][table]
    return tokens

# schema_info = SchemaInfo('data/spider/tables.json')
# print(schema_info.get_tokens())
