# Cracking the Code of AI-driven Inventory Decision

## Goal
1. Build a machine learning model that will suggest whether a part should be bought Make-to-Order (MTO) or Make-to-Stock (MTS) based on:
1. Part information (e.g. lead time, monthly demand, supplier info)
1. Traditional optimization model suggestion (MTO vs MTS)
1. Optional feedback (natural language suggestion describing justification or underlying rule supporting a correct decision)

## Constraints
1. Beats traditional optimization in accuracy
1. Can incorporate natural language feedback for new rules during run-time

## Full Design Doc
See [this design doc](https://docs.google.com/document/d/1XZCzf0UiKWMm1EEbJq63oaEKcUfz6U58dwCQwC0dwUM/edit?usp=sharing)
for the full details on models implemented.