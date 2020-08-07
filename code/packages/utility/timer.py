import datetime

class Timer():

	allTimer = {}

	def __init__(self, identifier = 'default'):
		Timer.allTimer[identifier]['start'] = None
		Timer.allTimer[identifier]['stop'] = None

	@staticmethod
	def start(identifier):
		Timer.allTimer[identifier] = {}
		Timer.allTimer[identifier]['start'] = datetime.datetime.now()


	@staticmethod
	def stop(identifier):
		Timer.allTimer[identifier]['stop'] = datetime.datetime.now()
		Timer.allTimer[identifier]['difference'] = Timer.getRequiredTime(identifier)

	@staticmethod
	def getRequiredTime(identifier):
		return Timer.getDifference(Timer.allTimer[identifier]['stop'], Timer.allTimer[identifier]['start'])

	@staticmethod
	def getDifference(dateTime1, dateTime2):
		elapsedTime = dateTime1 - dateTime2
		datetime.timedelta()
		return divmod(elapsedTime.total_seconds(), 60)

	@staticmethod
	def get():
		return datetime.datetime.now()

	@staticmethod
	def getFormattedTimers():
		formattedTimer = {}
		for identifier in Timer.allTimer.keys():
			if 'stop' not in Timer.allTimer[identifier].keys():
				continue
			formattedTimer[identifier] = {
				'start': Timer.getFormatedDate(Timer.allTimer[identifier]['start']),
				'stop': Timer.getFormatedDate(Timer.allTimer[identifier]['stop']),
				'difference': str(Timer.allTimer[identifier]['difference'])
			}
		return formattedTimer

	@staticmethod
	def getFormatedDate(datetimeValue = None):
		if not datetimeValue:
			datetimeValue = Timer.get()
		return datetimeValue.strftime("%y-%m-%d-%H-%M-%s")