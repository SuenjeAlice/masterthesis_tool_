
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random
import csv
import pandas

filename = "data/data.csv"
names = ['age', 'gender', 'species', 'mood', 'workethic', 'intelligence', 'luck', 'magic', 'careerfield', 'education', 'life-changing', 'events2', 'events', 'romance']
data = pandas.read_csv(filename, names=names, dtype=object, delimiter=";")
X = data.iloc[:, 0:-6]
y_career = data.iloc[:, -6]
y_education = data.iloc[:, -5]
y_lifechangingevents = data.iloc[:, -4]
y_events2 = data.iloc[:, -3]
y_events = data.iloc[:, -2]
y_romance = data.iloc[:, -1]

class Value_Assignment:

    # Decision tree
    # Values:
    def random_values(self):
        value1 = random.choice([0, 1]) #"compassionate", "cold"
        value2 = random.choice([1, 0]) #"ambitious", "lazy"
        value3 = random.choice([1, 0]) #"smart", "dumb"
        value4 = random.choice([1, 0]) #"lucky", "unlucky"

        return value1, value2, value3, value4

    def based_on_traits_values(self, traits):
        if traits["species"] == 1:
            value5 = 1
        elif traits["species"] == 2:
            value5 = 1
        elif traits["species"] == 3:
            value5 = 1
        else:
            value5 = 0
        return value5

    def decision_tree_values(self, traits):
        values = {}
        states = {}
        value1, value2, value3, value4 = self.random_values()
        value5 = self.based_on_traits_values(traits)
        values["mood"]= value1
        values["workethic"]=value2
        values["intelligence"] = value3
        values["luck"] = value4
        values["magic"] = value5

        X_current = []
        X_current.append(traits["age"])
        X_current.append(traits["gender"])
        X_current.append(traits["species"])
        X_current.append(values["mood"])
        X_current.append(values["workethic"])
        X_current.append(values["intelligence"])
        X_current.append(values["luck"])
        X_current.append(values["magic"])
        X_current = np.array(X_current)
        X_current = X_current.reshape(1, -1)

        state1 = self.decision_tree_career_field(traits, values, X_current)[0]
        state2 = self.decision_tree_education(traits, values, X_current)[0]
        state3 = self.decision_tree_life_changing_event(traits, values, X_current)[0]
        state4 = self.decision_tree_events2(traits, values, X_current)[0]
        state5 = self.decision_tree_events(traits, values, X_current)[0]
        state6 = self.decision_tree_romance(traits, values, X_current)[0]
        states["careerfield"]= int(state1)
        states["education"] = int(state2)
        states["life-changing"] = int(state3)
        states["events2"] = int(state4)
        states["events"] = int(state5)
        states["romance"] = int(state6)

        traits, values, states = self.values_into_words(traits, values, states)

        return traits, values, states

    def decision_tree_career_field(self, traits, values, X_current):
        dt = DecisionTreeClassifier()
        dt.fit(X.values, y_career.values)
        careerfields = dt.predict(X_current)
        return careerfields

    def decision_tree_education(self, traits, values, X_current):
        dt = DecisionTreeClassifier()
        dt.fit(X.values, y_education.values)
        education = dt.predict(X_current)
        return education

    def decision_tree_life_changing_event(self, traits, values, X_current):
        dt = DecisionTreeClassifier()
        dt.fit(X.values, y_lifechangingevents.values)
        lifechangingevents = dt.predict(X_current)
        return lifechangingevents

    def decision_tree_events2(self, traits, values, X_current):
        dt = DecisionTreeClassifier()
        dt.fit(X.values, y_events2.values)
        events2 = dt.predict(X_current)
        return events2

    def decision_tree_events(self, traits, values, X_current):
        dt = DecisionTreeClassifier()
        dt.fit(X.values, y_events.values)
        events = dt.predict(X_current)
        return events

    def decision_tree_romance(self, traits, values, X_current):
        dt = DecisionTreeClassifier()
        dt.fit(X.values, y_romance.values)
        romance = dt.predict(X_current)
        return romance

    def values_into_words(self, traits, values, states):
        #traits
        if traits["gender"] == 0:
            traits["gender"] = "non-binary"
        elif traits["gender"] == 1:
            traits["gender"] = "female"
        elif traits["gender"] == 2:
            traits["gender"] = "male"
        else:
            traits["gender"] = "female"

        if traits["species"] == 0:
            traits["species"] = "human"
        elif traits["species"] == 1:
            traits["species"] = "druid"
        elif traits["species"] == 2:
            traits["species"] = "mage"
        elif traits["species"] == 3:
            traits["species"] = "sorcerer"
        elif traits["species"] == 4:
            traits["species"] = "elf"
        elif traits["species"] == 5:
            traits["species"] = "dwarf"
        else:
            traits["species"] = "human"

        #values
        if values["mood"] == 0:
            values["mood"] = "compassionate"
        elif values["mood"] == 1:
            values["mood"] = "cold"

        if values["workethic"] == 0:
            values["workethic"] = "lazy"
        elif values["workethic"] == 1:
            values["workethic"] = "ambitious"

        if values["intelligence"] == 0:
            values["intelligence"] = "dumb"
        elif values["intelligence"] == 1:
            values["intelligence"] = "smart"

        if values["luck"] == 0:
            values["luck"] = "unlucky"
        elif values["luck"] == 1:
            values["luck"] = "lucky"

        if values["magic"] == 0:
            values["magic"] = "non-magical"
        elif values["magic"] == 1:
            values["magic"] = "magical"

        #states
        if states["careerfield"] == 0:
            states["careerfield"] = "none"
        elif states["careerfield"] == 1:
            states["careerfield"] = "physician"
        elif states["careerfield"] == 2:
            states["careerfield"] = "healer"
        elif states["careerfield"] == 3:
            states["careerfield"] = "soldier"
        elif states["careerfield"] == 4:
            states["careerfield"] = "teacher"
        elif states["careerfield"] == 5:
            states["careerfield"] = "bounty hunter"
        elif states["careerfield"] == 6:
            states["careerfield"] = "baker"
        elif states["careerfield"] == 7:
            states["careerfield"] = "alchemist"
        elif states["careerfield"] == 8:
            states["careerfield"] = "artisan"
        elif states["careerfield"] == 9:
            states["careerfield"] = "bookbinder"
        elif states["careerfield"] == 10:
            states["careerfield"] = "necromancer"
        elif states["careerfield"] == 11:
            states["careerfield"] = "blacksmith"
        elif states["careerfield"] == 12:
            states["careerfield"] = "scholar"

        if states["education"] == 0:
            states["education"] = "none"
        elif states["education"] == 1:
            states["education"] = "apprenticeship"
        elif states["education"] == 2:
            states["education"] = "university"
        elif states["education"] == 3:
            states["education"] = "learning on the go"

        if states["life-changing"] == 0:
            states["life-changing"] = "none"
        elif states["life-changing"] == 1:
            states["life-changing"] = "loss of a loved one"
        elif states["life-changing"] == 2:
            states["life-changing"] = "blessed by the gods"
        elif states["life-changing"] == 3:
            states["life-changing"] = "found a treasure"
        elif states["life-changing"] == 4:
            states["life-changing"] = "lost all savings"
        elif states["life-changing"] == 5:
            states["life-changing"] = "got deadly ill"
        elif states["life-changing"] == 6:
            states["life-changing"] = "watched a beheading"
        elif states["life-changing"] == 7:
            states["life-changing"] = "found a pot of gold"
        elif states["life-changing"] == 8:
            states["life-changing"] = "got knighted"

        if states["events2"] == 0:
            states["events2"] = "ate favourite food"
        elif states["events2"] == 1:
            states["events2"] = "had a wish come true"
        elif states["events2"] == 2:
            states["events2"] = "got embarrassed publicly"
        elif states["events2"] == 3:
            states["events2"] = "got robbed"
        elif states["events2"] == 4:
            states["events2"] = "went to the market"
        elif states["events2"] == 5:
            states["events2"] = "visited family"
        elif states["events2"] == 6:
            states["events2"] = "relaxed at a pond"
        elif states["events2"] == 7:
            states["events2"] = "learned something new"

        if states["events"] == 0:
            states["events"] = "found a lucky gold coin"
        elif states["events"] == 1:
            states["events"] = "broke an arm"
        elif states["events"] == 2:
            states["events"] = "made a new friend"
        elif states["events"] == 3:
            states["events"] = "got into a fight"
        elif states["events"] == 4:
            states["events"] = "ate cake"
        elif states["events"] == 5:
            states["events"] = "fell down a tree"
        elif states["events"] == 6:
            states["events"] = "stepped in horse-poo"
        elif states["events"] == 7:
            states["events"] = "received a gift"
        elif states["events"] == 8:
            states["events"] = "received a medal"

        if states["romance"] == 0:
            states["romance"] = "single"
        elif states["romance"] == 1:
            states["romance"] = "dating"
        elif states["romance"] == 2:
            states["romance"] = "in a relationship"
        elif states["romance"] == 3:
            states["romance"] = "married"
        elif states["romance"] == 4:
            states["romance"] = "married with kids"
        elif states["romance"] == 5:
            states["romance"] = "divorced"
        elif states["romance"] == 6:
            states["romance"] = "widowed"

        return traits, values, states


