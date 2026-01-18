import zlib
import json

#This is a light weight NBT parser.
#It just extracts the data from the NBT file, then returns the Numpy array for training.

f = open("./encoding.json")
encode_sequence = json.load(f)
f.close()
class analizer:
    def __init__(self, data: bytes,file:str):
        if data is None:
            with open(file, 'rb') as f:
                data = f.read()
                self.data = data
        else:
            self.data = data

    def get_chunk_data(self,x,z)->bytes:
        off = 4 * (x % 32 + z % 32 * 32)
        off = int.from_bytes(self.data[off : off + 3],byteorder='big')
        if off == 0 and self.data[off + 3] == 0:
            return None#this means the chunk at given coordinate has'n been generated yet
        off *= 4096
        length = int.from_bytes(self.data[off : off + 4],byteorder='big')
        if self.data[off + 4] == 1:
            raise BufferError("Gzip is not supported")
        return zlib.decompress(self.data[off + 5 : off + 5 + length - 1])

    def seek(data:bytes,string):
        '''
        search the data with given string
        desinged to avoid analize all the tags
        return : list of indexs 
        '''
        _range = len(data) - len(string)
        return [i for i in range(_range) if string.encode() == data[i:i+len(string)]]
