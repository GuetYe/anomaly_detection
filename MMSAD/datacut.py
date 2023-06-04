import os
filepath = "data.txt"
new_dataset_dir = "dataset"

if os.path.exists(new_dataset_dir) == False:
    # os.removedirs(new_dataset_dir)
    os.mkdir(new_dataset_dir)


file = open(filepath,"rb")

for line, l in enumerate(file):
    pass
#print("行数",line+1)
sensor_id = []
sensor_count = []
file.seek(0)
for count in range(line+1):
    linedata = file.readline()
    decode_data = linedata.decode("utf8").replace("\n","").split(" ")
    if decode_data[0] == "2004-03-06" or decode_data[0] == "2004-03-05" or decode_data[0] == "2004-03-04" or decode_data[0] == "2004-03-07":
        if len(decode_data) != 8:
            writer = open(new_dataset_dir + "/" + "erro.txt", "ab")
            writer.write(linedata)
            writer.close()
        else:
            if decode_data[3] not in sensor_id:
                sensor_id.append(decode_data[3])
                sensor_count.append(0)
            sensor_count[sensor_id.index(decode_data[3])] = sensor_count[sensor_id.index(decode_data[3])] + 1
            print(sensor_count)
            writer = open(new_dataset_dir + "/" + str(decode_data[3])+".txt","ab")
            writer.write(linedata)
            writer.close()
    else:
        continue
file.close()

countfile = open("count.txt","wb")
countfile.write((str(sensor_id)+ "\n" +str(sensor_count)).encode("utf8"))
