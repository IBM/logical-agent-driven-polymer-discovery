
import dill as pickle

def get_property(data, parameters):
    try:
        return data[parameters]
    except IndexError:
        return None

if __name__ == '__main__':

	f = open("./data/polymerDiscovery/train/function_1.pkl","wb")
	pickle.dump(get_property, f)
	f.close()

	f = open("./data/polymerDiscovery/test/function_1.pkl","wb")
	pickle.dump(get_property, f)
	f.close()