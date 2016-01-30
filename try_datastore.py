from livestockwatch.data_store import LivestockStorage
from livestockwatch.config import DCT_CATTLE_ID

l_store = LivestockStorage()
# cattle_id = "860053727"

# l_store.update_ca1scnd(cattle_id)

# TA01_ID = ["608", "610", "613", "614", "615", "617"]
# for cattle_id in TA01_ID:
#     l_store.update_vitamin_a(cattle_id)

# l_store.recalculate_ca("WA02_2015_10_31_14_16_7_689035")
# l_store.recalculate_cs("WA02_2015_10_31_14_16_7_689035")

# for k,v in DCT_CATTLE_ID.items():
#     print("key: {} value: {}".format(k, v))
#     l_store.update_cattle_id(k, v)

# l_store.add_new_cs_column("is_valid", True)

# l_store.mark_invalid_cs("/Volumes/fitramhd/BISE/dataset-final/false_dataset.xlsx")


# l_store.update_vitamin_a()

# l_store.generate_cs1sec()
# l_store.gen_cs1sec_datetaken()

l_store.gen_cs1secfirstframe()
l_store.gen_cs1secfirstframe_datetaken()