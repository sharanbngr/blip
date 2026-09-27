from hypothesis import settings

settings.register_profile("no_deadline", deadline=None)
settings.load_profile("no_deadline")
