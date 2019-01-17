
class node:
	def __init__(self,name,positive,negative):
		self.name= name
		self.positive=positive
		self.negative=negative
		self.children ={}   # {low:pointer to low,high:pointer to high}

	def insert(self,obj):
		self.children.append(obj)

	