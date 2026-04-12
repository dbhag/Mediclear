## Results

| Task                        | Metric   | Score | Explanation                                                                                                              |
| --------------------------- | -------- | ----- | ------------------------------------------------------------------------------------------------------------------------ |
| Medical Text Simplification | BLEU     |  3.60 | The BLEU score is low, which means the model output does not match the reference wording closely.                        |
| Medical Text Simplification | SARI     | 35.28 | The SARI score shows the model is doing some useful simplification by changing, removing, and keeping parts of the text. |
| Credibility Classification  | Accuracy |  0.56 | The model predicted the correct label for 56% of the test examples.                                                      |
| Credibility Classification  | Macro-F1 |  0.31 | The lower Macro-F1 suggests the model does not perform equally well on all classes.                                      |


Explanation

Overall, MediClear works as a basic end-to-end system for both simplification and credibility prediction. The results show that the project is functional, but both models can still be improved with more tuning, better training data, and further testing.