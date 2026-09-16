# ``DataLens``

Local-regression toolkit in pure Swift: Cleveland-style LOESS first, then
clean-room Loader-style adaptive smoothing and local likelihood.

## Overview

```swift
import DataLens
let xs = (0..<15).map { [Double($0)] }
let ys = xs.map { 2 * $0[0] - 1 }
let fit = Loess.fit(trainX: xs, trainY: ys, span: 0.5, degree: 1)!
fit.predict([7.5]) // 14.0
```

## Topics

### Smoothing

- ``Loess``
- ``LoessWeight``
- ``NadarayaWatson``
- ``WhittakerEilers``
- ``TotalVariation``
- ``AdaptiveLoess``
- ``LocalLikelihood``
- ``LocalLikelihoodFamily``
- ``AutomaticSmoother``
- ``FittedSmoother``
- ``TuningSummary``
- ``ExtrapolationPolicy``
- ``DataLens``
