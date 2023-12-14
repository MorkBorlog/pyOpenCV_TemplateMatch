
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

gesicht = cv.imread(r'waldo-gesicht.jpg', 0)
leinwand = cv.imread(r'waldo-grossesbild.jpg')
leinwand_grau = cv.imread(r'waldo-grossesbild.jpg', 0)

# Wenn die Farben nicht convertiert werden, dann ist das Ausgabebild kein RGB
leinwand = cv.cvtColor(leinwand, cv.COLOR_BGR2RGB)

breite, hoehe = gesicht.shape[::-1]

ergebnis = cv.matchTemplate(gesicht, leinwand_grau, cv.TM_CCOEFF_NORMED)

min_wert, max_wert, min_position, max_position = cv.minMaxLoc(ergebnis)

oben_links = max_position
unten_rechts = (oben_links[0] + breite, oben_links[1] + hoehe)

cv.rectangle(leinwand, oben_links, unten_rechts, (0,255,0), 2)
cv.putText(leinwand, 'Waldo', unten_rechts, cv.FONT_HERSHEY_SIMPLEX, 5, (255,0,0), 9)

plt.imshow(leinwand)
plt.suptitle('Ergebnis')
plt.show()

