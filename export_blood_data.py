import xlrd
import re
from collections import OrderedDict
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from odo import odo, discover, resource
import pymongo
from bokeh.plotting import figure, output_server, show, output_file, save

import matplotlib.pyplot as plt
from livestockwatch import config

from IPython import embed
from pprint import pprint

connection = pymongo.MongoClient(config.MONGO_URL)
db = connection.livestockwatch
blood_data = db.blood_data
blood_data_interp = db.blood_data_interp

cattle_basic_datapath = os.path.join("cattle_excel_data", "basic_data.xlsx")
wa_blood_datapath = os.path.join("cattle_excel_data", "wadayama_vitA.xlsx")
tk_blood_datapath = os.path.join("cattle_excel_data", "takahara_vitA.xlsx")


def export_TK_blood_data(excel_path):
    connection = pymongo.MongoClient(config.MONGO_URL)
    db = connection.livestockwatch
    blood_data = db.blood_data
    blood_excel = pd.read_excel(excel_path)
    new_names = ["datetaken", "cattle_id", "vit_a_ug_ml", "vit_a", "beta_caroten", "vit_e"]
    blood_excel.rename(columns=dict(zip(blood_excel.columns, new_names)), inplace=True)
    blood_excel.dropna(inplace=True)
    for i in blood_excel.index:
        datetaken = blood_excel.ix[i].datetaken.to_datetime()
        cattle_id = re.findall(r'\d+', str(blood_excel.ix[i].cattle_id))[0]
        _id = "{}_{}_{}_{}".format(
                cattle_id,
                str(datetaken.year),
                str(datetaken.month),
                str(datetaken.day)
        )
        dct_data = OrderedDict([
            ("_id", _id),
            ("cattle_id", cattle_id),
            ("vit_a", blood_excel.ix[i].vit_a),
            ("beta_caroten", blood_excel.ix[i].beta_caroten),
            ("vit_e", blood_excel.ix[i].vit_e),
            ("datetaken", datetaken),
        ])
        pprint(dct_data)
        retry = True
        while retry:
            try:
                blood_data.update_one(
                        {"_id": _id},
                        {"$set": dct_data},
                        upsert=True
                )
                retry = False
            except NameError as e:
                print("[BREAK], {} {}".format(type(e), e))
                retry = False
            except Exception as e:
                print("[RETRY], {} {}".format(type(e), e))
                connection.close()
                connection = pymongo.MongoClient(config.MONGO_URL)
                db = connection.livestockwatch
                blood_data = db.blood_data
                retry = True


def export_WA_blood_data(excel_path):
    connection = pymongo.MongoClient(config.MONGO_URL)
    db = connection.livestockwatch
    blood_data = db.blood_data
    lst_data_frame = []
    for i in range(28):
        try:
            blood_excel = pd.read_excel(excel_path, sheetname=i)
            new_names = ["cattle_id", "vit_a", "beta_caroten", "vit_e", "datetaken"]
            blood_excel.rename(columns=dict(zip(blood_excel.columns, new_names)), inplace=True)
            blood_excel.dropna(inplace=True)
            if blood_excel.empty:
                print('DataFrame {} is empty!'.format(i+1))
            else:
                print("DataFrame {} exists".format(i+1))
                lst_data_frame.append(blood_excel)
        except xlrd.XLRDError as e:
            print("Error type: {}, {}".format(type(e), e))

    for df in lst_data_frame:
        for i in df.index:
            datetaken = df.ix[i].datetaken.to_datetime()
            cattle_id = str(df.ix[i].cattle_id)
            _id = "{}_{}_{}_{}".format(
                    cattle_id,
                    str(datetaken.year),
                    str(datetaken.month),
                    str(datetaken.day)
            )
            dct_data = OrderedDict([
                # ("_id", _id),
                ("cattle_id", cattle_id),
                ("vit_a", df.ix[i].vit_a),
                ("beta_caroten", df.ix[i].beta_caroten),
                ("vit_e", df.ix[i].vit_e),
                ("datetaken", datetaken),
            ])
            pprint(dct_data)
            retry = True
            while retry:
                try:
                    blood_data.update_one(
                            {"_id": _id},
                            {"$set": dct_data},
                            upsert=True
                    )
                    retry = False
                except NameError as e:
                    print("[BREAK], {} {}".format(type(e), e))
                    retry = False
                except Exception as e:
                    print("[RETRY], {} {}".format(type(e), e))
                    connection.close()
                    connection = pymongo.MongoClient(config.MONGO_URL)
                    db = connection.livestockwatch
                    blood_data = db.blood_data
                    retry = True


def interpolate_blood_data():
    connection = pymongo.MongoClient(config.MONGO_URL)
    db = connection.livestockwatch
    blood_data_interp = db.blood_data_interp
    blood_data_interp.create_index([("cattle_id", pymongo.ASCENDING)])
    blood_data_interp.create_index([
        ("cattle_id", pymongo.ASCENDING),
        ("vit_a", pymongo.DESCENDING)
    ])

    date_x = np.arange(datetime(2015, 1, 1),datetime(2015,12,27), timedelta(days=1)).astype(datetime)

    bd_resource = resource("{}::blood_data".format(config.MONGO_URL))
    ds_blood_data = "758 * {cattle_id: string, datetaken: datetime, vit_a: float64, beta_caroten: float64, vit_e: float64}"
    blood_df = odo("{}::blood_data".format(config.MONGO_URL), pd.DataFrame, dshape=ds_blood_data)
    lst_cattle_id = blood_df.cattle_id.unique()
    for cattle_id in lst_cattle_id:
    # for cattle_id in ["615"]:
        vita_series = pd.Series(blood_df[blood_df.cattle_id == cattle_id].set_index("datetaken").to_dict()["vit_a"],
                                 index=date_x)
        vita_interp = vita_series.interpolate(method="linear")

        vite_series = pd.Series(blood_df[blood_df.cattle_id == cattle_id].set_index("datetaken").to_dict()["vit_e"],
                                 index=date_x)
        vite_interp = vite_series.interpolate(method="linear")

        beta_caroten_series = pd.Series(blood_df[blood_df.cattle_id == cattle_id].set_index("datetaken").to_dict()["beta_caroten"],
                                 index=date_x)
        beta_caroten_interp = beta_caroten_series.interpolate(method="linear")

        p = figure(tools=['xwheel_zoom'], width=1000, height=800, x_axis_type="datetime")

        p.circle(vita_series.index,vita_series, size=10, color="red", alpha=1)
        p.line(vita_interp.index, vita_interp, color="salmon", line_width=3, legend="Vitamin A")

        # p.circle(vite_series.index,vite_series, size=10, color="green", alpha=1)
        # p.line(vite_interp.index, vite_interp, color="MediumSeaGreen", line_width=3, legend="Vitamin E")
        #
        # p.circle(beta_caroten_series.index,beta_caroten_series, size=10, color="blue", alpha=1)
        # p.line(beta_caroten_interp.index, beta_caroten_interp, color="DodgerBlue", line_width=3, legend="Beta Carotene")

        p.xaxis.axis_label = "Date Taken"
        p.xaxis.axis_line_width = 1

        p.yaxis.axis_label = "Vitamin A(IU/DL)"

        output_path = os.path.join(os.getcwd(),"vita_plot", "{}_vita.html".format(cattle_id))
        print(output_path)
        output_file(output_path, title="Vita cattle_id:{}".format(cattle_id))
        save(p)
        for datetaken in vita_interp.index:
            _id = "{}_{}_{}_{}".format(
                    cattle_id,
                    str(datetaken.year),
                    str(datetaken.month),
                    str(datetaken.day)
            )
            dct_data = OrderedDict([
                # ("_id", _id),
                ("cattle_id", cattle_id),
                ("vit_a", vita_interp[datetaken]),
                ("beta_caroten", beta_caroten_interp[datetaken]),
                ("vit_e", vite_interp[datetaken]),
                ("datetaken", datetaken),
            ])
            pprint(_id)
            retry = True
            while retry:
                try:
                    blood_data_interp.update_one(
                            {"_id": _id},
                            {"$set": dct_data},
                            upsert=True
                    )
                    retry = False
                except NameError as e:
                    print("[BREAK], {} {}".format(type(e), e))
                    retry = False
                except Exception as e:
                    print("[RETRY], {} {}".format(type(e), e))
                    connection.close()
                    connection = pymongo.MongoClient(config.MONGO_URL)
                    db = connection.livestockwatch
                    blood_data_interp = db.blood_data_interp
                    retry = True

if __name__ == '__main__':
    # blood_data.drop()
    # blood_data.create_index([("cattle_id", pymongo.ASCENDING)])
    # blood_data.create_index([("_id", pymongo.ASCENDING)], unique=True)
    # export_TK_blood_data(tk_blood_datapath)
    # export_WA_blood_data(wa_blood_datapath)
    # blood_data_interp.drop()
    interpolate_blood_data()
    # blood_data.reindex()