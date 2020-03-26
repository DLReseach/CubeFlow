import sqlite3
import pickle
from pathlib import Path


def fetch_set(set_type, sql_path, masks_path):
	sql_file = sql_path.joinpath(set_type + '_transformed.db')
	mask_file = masks_path.joinpath(set_type + '_set.pickle')
	con = sqlite3.connect(sql_file)
	cur = con.cursor()
	query = 'SELECT event_no FROM meta'
	cur.execute(query)
	fetched_set = cur.fetchall()
	set_list = [event[0] for event in fetched_set]
	with open(mask_file, 'wb') as f:
		pickle.dump(set_list, f)


sql_path = Path().home().joinpath('CubeFlowData').joinpath('db')
masks_path = Path().home().joinpath('CubeFlowData').joinpath('masks')
sets = ['test']
for set_type in sets:
	fetch_set(set_type, sql_path, masks_path)
