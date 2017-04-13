import _pickle as pickle
import numpy as np

dataPath = "../data/"
logPath = "../log/"

ipinyouPath = dataPath + "ipinyou-data/"
#olaPath = dataPath + "ola-data/"


ipinyou_camps = ["1458", "2259", "2261", "2821", "2997", "3358", "3386", "3427", "3476"]
# ola_camps = ["57f4c60d4366094103939190", "57f5e35e4366094103939198", "57f5e2484366094103939193",
# 			 "57f5e2524366094103939194", "57f5e2594366094103939195","57f350ad4366090a002ed6b8",
# 			 "57f7667b43660948da6becc7", "57f7667643660948da6becc6", "57f7668843660948da6becc8"]

ipinyou_max_market_price = 300
ola_max_market_price = 300

info_keys = ["imp_test", "cost_test", "clk_test", "imp_train", "cost_train", "clk_train", "field", "dim", "price_counter_train"]

def get_camp_info(camp, src="ipinyou"):
	if src == "ipinyou":
		info = pickle.load(open(ipinyouPath + camp + "/info.txt", "rb"))
		print (info)
	elif src == "ola":
		info = pickle.load(open(olaPath + camp + "/info.txt", "rb"),encoding='latin1')
		print (info)
	return info