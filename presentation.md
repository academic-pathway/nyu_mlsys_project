---
marp: true
theme: codecafe
class: invert
footer: MLSYS Presentation, SP24 [](https://code-cafe.nl)
--- 

# PyTorch Distributed: <br>Experiences on Accelerating Data Parallel Training

<br>
<br>
Sanho Lee (shl8607), Euijae Kim (ek3955)

<!-- paginate: true -->

--- 

<!-- ## TEST (THIS WON'T BE INCLUDED)

- E-mail mij op noah.beij@code-cafe.nl
- Join de CodeCafé-community op Discord!

![bg right 70%](https://assets.nbeij.nl/marp/assets/codecafe.png) -->

## 1. Motivation

- Distributed Training
  - Recent advances in deep learning argue for the value of large datasets and large models, which necessitates the ability to scale out model training to more computational resources.
- Large datasets and large models
- asdf

---

## 2. Problem Definition

- Mathematical equivalence
- Non-intrusive and interceptive API
- High performance

---

## 3. Main Contribution of work

- Design and implementation of a widely adopted industrial state-of-the-art distributed training solution
- Real-world caveats that were overlooked by prior work
- Performance tuning experiences collected from users and summarized several directions for future improvements

---

## 4. Central Design/Idea (API)



--- 

## 4. Central Design/Idea (Naive Solution of Gradient Reduction)

```python
import torch
```

---

## 4. Central Design/Idea (Gradient Bucketing)

![70%](./gradient_bucketing.png)

---

## 6. Related Work

- https://arxiv.org/pdf/2309.06497.pdf / what are the most closely related other systems/results? how are they similar? how are they different? is the difference between the work you are presenting and the related work significant? / we expect you to find and read the major papers cited by the authors as related (you don’t need to read them in as much detail as the paper you are presenting! but in enough detail to understand how the work they describe differs from that in the paper you are presenting.)
- we expect you to search for more recent related work published after (or perhaps simultaneously with) the paper you are presenting – no need to claim the work you are presenting is “better” or “worse” than a particular piece of related work (though you may of course do so if you feel that way!); often it is simply that the two pieces of work are different–but you must articulate the precise difference (e.g., “these other authors solve a slightly different problem...”) / For each class, we often have presentations on two related papers. You are welcome to discuss related work with the team of the other presentation.
