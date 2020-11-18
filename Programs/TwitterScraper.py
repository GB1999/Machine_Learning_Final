import twint

c = twint.Config()
c.Search = "INTC"
c.Min_likes = 5
twint.run.Search(c)
print(c)