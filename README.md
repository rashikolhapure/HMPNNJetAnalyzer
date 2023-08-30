# 🚀 HyperGraph Magic: Unraveling Jets with HMPNNs


📡 **"Houston, we have jets!"** 🛰️

🎉 Welcome to the journey of **HyperGraph Message Passing Neural Networks** (HMPNNs) exploring the mesmerizing realm of high-energy particle collisions! Strap in, because we're about to decode the symphony of particles through a fusion of physics and machine learning. 🤯

![Particle Orchestra](images/atlas.png)

## ⚛️ Quantum Chronicles of QCD

🔬 **Quantum Chromodynamics (QCD)**, the tale of strong quark-gluon interactions, sets the stage for our journey:

https://github.com/rajveer43/hep_ml/assets/64583161/c6034ad6-aa9b-4c5b-a00b-fe305d05cbbd

🔥 **"Particle Aria"** - Jets emerge as sparkling sprays of particles in high-energy collisions, their songs resonating through the cosmos. 🌠

⚛️ **"QCD's Cosmic Dance"** - QCD jets, birthed from quarks and gluons, converse a different cosmic language compared to non-QCD jets. Their energy, multiplicity, and dance steps set them apart. 💃

🪐 **"Quantum Properties"** - Jet classification unveils their essence: the jet mass, substructure, energy, and a plethora of features shape their cosmic choreography. 🛰️

## 🌟 Stars in the Jet Constellation


<!--add the dataset of the particle jets -->

🌟 **"Our Celestial Ensemble"** - The dataset twinkles with MC simulated events, unraveling top quark tagging mysteries. 1.2M training events, 400k validation events, and 400k test events make up our cosmic ensemble. 🎭

🔍 **"Particle Puzzle Pieces"** - The dataset embodies hadronic tops for the signal, QCD diets background, Pythia8's ATLAS detector card, and the Pythia 8-generated quark and gluon jet datasets. Each piece holds a cosmic puzzle. 🧩

## 🚀 Particle Jet Portals

🪐 **"Hypergraph Voyage"** - We steer the cosmic ship of Hypergraph Message Passing, crafting graphs in the (η, φ)-plane and passing cosmic messages to weave cosmic insights. 🚢

🎩 **"Magic Four-Vectors"** - Directions sculpt our message weights, as four-vectors dance with information exchange, painting the cosmic symphony of jet features. 🌈

🧙‍♂️ **"Neural Cosmic Oracle"** - A cosmic climax ensues as our cosmic representation reaches the cosmic Neural Network Oracle. The oracle's verdict unveils the cosmic binary classification score, decoding QCD and non-QCD jets' cosmic essence. 🌌

## 🪐 Pioneering Particle Performances

In the grand theater of high-energy collisions, the particle jets dance with complexity and mystery. These performances are captivating, yes, but often a conundrum. Fear not, for our cast of **Neural Networks** are here to decipher the enigmatic jets with precision! 🌌

![irc](images/auc.jpg)

🤖 **"Classifier Extraordinaire!"** - Our algorithm shines in distinguishing jet types, unraveling secrets essential for a myriad of physics analyses. 🕵️‍♂️

![irc](images/ir1.jpg)

🧙‍♂️ **"Magic of IRC Safety!"** - With a sprinkle of physics, our HMPNNs honor IRC safety, ensuring predictions stay steadfast even in the face of soft emissions. 🪄

![qcd_radiation](images/energy%20comaprison.png)

📡 **"Navigating the Jet Stream!"** - Taming the QCD radiation dragon, our neural wizards stay cool while venturing into the vast jet landscape. 🌀

![different size of qcd jets](images/Ogul_TCT_Jets_Tracks_HighPTj.png)

💥 **"Scalable Sorcery!"** - Be it tiny jets or colossal ones, our HMPNNs flaunt the magic of scalability, adapting to various jet sizes and types. 🎩

## 🌟 Glorious Galaxy of Findings

![AUC Curve](images/auc.jpg)

✨ **Stellar Performance** - Our model shines bright with stellar accuracy and AUC in distinguishing top quarks from QCD jets. The stars align for precision! ✨

### AUC Values for Gluons vs Quark Tagging Dataset
<table>
  <tr>
    <th>Sr. No.</th>
    <th>R<sub>0</sub></th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>1</td>
    <td>0.1</td>
    <td>0.8824±0.0005</td>
  </tr>
  <tr>
    <td>2</td>
    <td>0.1</td>
    <td>0.8888±0.0013</td>
  </tr>
  <tr>
    <td>3</td>
    <td>0.2</td>
    <td>0.8909±0.0009</td>
  </tr>
  <tr>
    <td>4</td>
    <td>0.3</td>
    <td>0.8916±0.0008</td>
  </tr>
  <tr>
    <td>5</td>
    <td>0.4</td>
    <td>0.8919±0.0006</td>
  </tr>
</table>

### AUC Values for Top Tagging Dataset

<table>
  <tr>
    <th>Sr. No.</th>
    <th>R<sub>0</sub></th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>1</td>
    <td>0.1</td>
    <td>0.9734±0.0009</td>
  </tr>
  <tr>
    <td>2</td>
    <td>0.2</td>
    <td>0.9764±0.0004</td>
  </tr>
  <tr>
    <td>3</td>
    <td>0.3</td>
    <td>0.9779±0.0005</td>
  </tr>
  <tr>
    <td>4</td>
    <td>0.4</td>
    <td>0.9782±0.0002</td>
  </tr>
  <tr>
    <td>5</td>
    <td>0.5</td>
    <td>0.9781±0.0002</td>
  </tr>
</table>

### AUC Values for W Tagging Dataset

<table>
  <tr>
    <th>Sr. No.</th>
    <th>R<sub>0</sub></th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>1</td>
    <td>0.1</td>
    <td>0.9865±0.0004</td>
  </tr>
  <tr>
    <td>2</td>
    <td>0.2</td>
    <td>0.9864±0.0004</td>
  </tr>
  <tr>
    <td>3</td>
    <td>0.3</td>
    <td>0.9863±0.0004</td>
  </tr>
  <tr>
    <td>4</td>
    <td>0.4</td>
    <td>0.9868±0.0004</td>
  </tr>
  <tr>
    <td>5</td>
    <td>0.5</td>
    <td>0.9868±0.0005</td>
  </tr>
</table>

![Permutation Invarient](images/PI.png)

🌐 **Hypergraph Odyssey** - In the land of hypergraphs, our HMPNNs are fearless explorers, traversing multiple nodes, hyperedges, and labels. 🛰️

![AUC Curve](images/roc.jpg)

🌌 **Cosmic Radius Reckoning** - Tune in for the cosmic dance as we test different values of R, controlling jet radius. Bigger isn't always better, and smaller isn't always wiser. 🔮

## 🚀 Quantum Implications

From deep within the heart of high-energy physics to the cosmos of machine learning, our journey opens realms of possibility:

- 🌌 **Jet Pioneering**: Elevate QCD and non-QCD jet classification for enhanced high-energy physics experiments. Results that are out of this world! 🚀

- 💡 **Collider Enchantment**: Enrich collider event simulations with precise jet classification. It's like a magical touch to the particle orchestra. 🎻

- 🌀 **Anomaly Alchemy**: Detect anomalies in collider data and unlock the secrets of new physics beyond the Standard Model. Spells of discovery are cast! 🔍

- 📈 **Calibration Chronicles**: Jet calibration gets a boost with the wizardry of HMPNNs, ensuring particle property determinations are on point. 📊

- 🎩 **Innovation Spells**: Our HMPNN saga inspires new machine learning techniques, reverberating beyond particle physics into diverse realms. 💬

<!-- ![End of the Voyage](project-end-image.jpg) -->

## 🌌 Beam Me Up, Scotty!

1. 🛸 Clone the repository: `git clone https://github.com/rajveer43/hep_ml.git`
2. 🌌 Navigate to the project realm: `cd hep_ml`
3. 🪄 Set up your mystical environment: `pip install -r requirements.txt`
4. 🌟 Explore the `images/` galaxy for captivating data explorations and spellbinding model training examples.

## 🪐 Beyond the Stars: Cosmic Fellowship

- **Join the Cosmic Circle**: A cosmic contribution is a beacon in our galactic journey. Engage through a cosmic pull request and intertwine your cosmic magic!
- **Astro-Potion (Issue)**: For grand cosmic spells, step into the cosmic realm of issues to conjure discussions on cosmic ideas.

## ⚡ Cosmic Code

This cosmic journey abides under the [Cosmic License](LICENSE), granting cosmic sovereignty in wielding its magic!

## ⚡ Warp-Speed License

This project is enchanted under the [MIT License](LICENSE), allowing you to wield its powers with freedom!

<!-- --- -->

<!-- 🚀 Embark with us on a journey that transcends particles and codes, and let's redefine the frontiers of understanding together! ⚛️ -->


<!-- # 🚀 HyperGraph Alchemy: Deciphering Particle Jets with HMPNNs

🌌 **"Where Particles Meet Magic!"** 🪄

🎉 Greetings, cosmic wanderer! Venture into the cosmic depths with us as we wield the power of **HyperGraph Message Passing Neural Networks (HMPNNs)** to illuminate the enigmatic realm of high-energy particle collisions. 🌠 -->

<!-- ![Cosmic Symphony](project-banner-image.jpg) -->




---

⚛️ Venture forth with us into the cosmic frontier where particles dance and codes merge, reshaping the cosmos of understanding! 🌌
