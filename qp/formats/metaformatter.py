class MetaFormatter(type):
    def __init__(cls, name, bases, dct):
        print ('calling MetaFormatter.init() -- ' + str(cls))
        print(name, bases, dct)
        cls.name = name.lower()
        cls.fundamentals = ['cdf', 'fit', 'pdf', 'ppf', 'rvs']
        if not hasattr(cls, 'registry'):
            cls.registry = {}
            cls.graph = {}
            for i in cls.fundamentals:
                cls.graph[i] = []
        else:
            format_id = cls.name
            cls.registry[format_id] = cls
            cls.graph[format_id] = []
        cls._set_easy()
        cls._build_graph()
        super(MetaFormatter, cls).__init__(name, bases, dct)
