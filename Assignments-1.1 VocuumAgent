import random

class Environment(object):

    def __init__(self):

        self.locationCondition = {
            'A': random.randint(0, 1),
            'B': random.randint(0, 1)
        }

        self.dustLevel = {
            'A': random.randint(0, 3),
            'B': random.randint(0, 3)
        }

        self.wetFloor = {
            'A': random.randint(0, 1),
            'B': random.randint(0, 1)
        }

        self.chair = {
            'A': random.randint(0, 1),
            'B': random.randint(0, 1)
        }


class SimpleReflexVacuumAgent(Environment):

    def __init__(self, Environment):

        print("\nInitial Environment")
        print("Location Condition:", Environment.locationCondition)
        print("Dust Level:", Environment.dustLevel)
        print("Wet Floor:", Environment.wetFloor)
        print("Chair:", Environment.chair)

        Score = 0

        vacuumLocation = random.randint(0, 1)

        if vacuumLocation == 0:
            currentLocation = 'A'
            otherLocation = 'B'
        else:
            currentLocation = 'B'
            otherLocation = 'A'

        print("\nVacuum is at Location", currentLocation)

        result = self.check_location(Environment, currentLocation)

        if result == "SAFE":

            if Environment.locationCondition[currentLocation] == 1:

                print("Location", currentLocation, "is Dirty.")

                dust = Environment.dustLevel[currentLocation]

                print("Dust Level:", dust)

                if dust == 1:
                    print("Suction Power: LOW")

                elif dust == 2:
                    print("Suction Power: MEDIUM")

                elif dust == 3:
                    print("Suction Power: HIGH")

                Environment.locationCondition[currentLocation] = 0
                Environment.dustLevel[currentLocation] = 0

                Score += 1

                print("Location", currentLocation, "has been Cleaned.")

            else:

                print("Location", currentLocation, "is already Clean.")

        else:

            print("Location", currentLocation, "is unsafe.")
            print("Vacuum will avoid this location.")

        print("\nChecking Location", otherLocation)

        result = self.check_location(Environment, otherLocation)

        if result == "SAFE":

            print("Moving to Location", otherLocation)

            Score -= 1

            if Environment.locationCondition[otherLocation] == 1:

                print("Location", otherLocation, "is Dirty.")

                dust = Environment.dustLevel[otherLocation]

                print("Dust Level:", dust)

                if dust == 1:
                    print("Suction Power: LOW")

                elif dust == 2:
                    print("Suction Power: MEDIUM")

                elif dust == 3:
                    print("Suction Power: HIGH")

                Environment.locationCondition[otherLocation] = 0
                Environment.dustLevel[otherLocation] = 0

                Score += 1

                print("Location", otherLocation, "has been Cleaned.")

            else:

                print("Location", otherLocation, "is already Clean.")

        else:

            print("Location", otherLocation, "is unsafe.")
            print("Vacuum will avoid this location.")

        print("\nFinal Environment")
        print("Location Condition:", Environment.locationCondition)
        print("Dust Level:", Environment.dustLevel)
        print("Wet Floor:", Environment.wetFloor)
        print("Chair:", Environment.chair)

        print("\nPerformance Measurement:", Score)

    def check_location(self, Environment, location):

        if Environment.wetFloor[location] == 1:

            print("\nWet floor detected at Location", location)

            return "UNSAFE"

        if Environment.chair[location] == 1:

            print("\nChair detected at Location", location)

            return "UNSAFE"

        print("Location is safe.")

        return "SAFE"


theEnvironment = Environment()
theVacuum = SimpleReflexVacuumAgent(theEnvironment)
