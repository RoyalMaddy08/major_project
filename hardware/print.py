print("float data[500][4] = {")
for row in data_slice:
    print("  {%.3f, %.3f, %.3f, %d}," % (
        row[0], row[1], row[2], int(row[3])
    ))
print("};")
