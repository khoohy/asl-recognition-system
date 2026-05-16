# Real-Time American Sign Language Recognition and Translation System Report

## Abstract

This report presents the development, implementation, and evaluation of a real-time American Sign Language recognition and translation system designed to operate on standard consumer hardware. The project addresses a persistent accessibility problem: deaf and hard-of-hearing individuals often face communication barriers when interacting with hearing people in situations where a human interpreter is unavailable. Existing technological approaches to sign language processing have demonstrated important progress, but many remain limited by one or more practical constraints, including dependency on specialized hardware, high computational cost, restricted vocabularies, limited real-time performance, or lack of direct spoken output (Alsharif et al., 2025; Gan et al., 2023; Holmes et al., 2024; Tan et al., 2024). In response to this gap, the present project develops a webcam-based system that performs isolated sign recognition over a 300-word ASL vocabulary derived from WLASL, using MediaPipe landmark extraction, temporal sequence modelling, prediction stabilisation, and text-to-speech output.

The report covers the project as a whole rather than only a single evaluation phase. It explains the problem statement, the literature-informed design choices, the system architecture, the dataset and preprocessing pipeline, the deep learning model, the real-time inference pipeline, and the evaluation strategy used to assess both technical performance and practical usability. The methodology combines dataset-driven model training with deployment-oriented engineering. Keypoint normalization, missing-keypoint handling, sign-focused feature selection, temporal buffering, and class-specific runtime controls were introduced to maintain consistency between training and inference while preserving real-time responsiveness (De Coster et al., 2023; Holmes et al., 2024; Tan et al., 2024). A BiLSTM-based temporal classifier with attention was used as the principal recognition backbone, supported by deployment logic such as confidence gating, confusion-pair suppression, motion requirements for unstable signs, and optional speech synthesis (Gan et al., 2023; Kamble, 2025; Uddin et al., 2025).

The results demonstrate that the project achieved a functioning end-to-end prototype capable of real-time ASL sign recognition and translation on a mid-sized vocabulary without requiring depth sensors, motion gloves, multi-camera rigs, or server-class hardware. Held-out evaluation produced overall sample-level accuracy of `51.95%`, with substantially stronger Top-5 behaviour across the stronger experiment variants. Among the main experiment stages, validation Top-1 performance reached as high as `72.31%`, validation Top-5 reached `89.33%`, held-out test Top-1 reached `65.01%`, and held-out test Top-5 reached `87.59%`. Most importantly, live system testing across the 300-sign vocabulary showed that iterative refinement improved operational success from `82.7%` to `91.67%`, demonstrating strong practical viability even though a subset of difficult glosses remained sensitive to temporal thresholds, motion gating, and confusion with visually similar signs. Overall, the project confirms that a lightweight, keypoint-based, real-time ASL recogniser with integrated speech output is feasible on standard hardware and represents a meaningful step toward more accessible assistive communication technology.

**Keywords:** American Sign Language; real-time recognition; MediaPipe landmarks; WLASL300; sign-to-speech

## CHAPTER 1 INTRODUCTION

Communication accessibility remains one of the most important challenges in inclusive human-computer interaction. For deaf and hard-of-hearing individuals, communication with hearing people can become difficult in everyday settings when a shared signed language is absent and when human sign-language interpreters are not available. Although written text, lip reading, and mobile messaging can partially bridge this gap, these methods are not always natural, immediate, or sufficient. A real-time sign-language recognition and translation system that can operate with widely available hardware offers a promising assistive pathway because it has the potential to convert signs into readable and audible output without requiring costly equipment or highly controlled environments (Alsharif et al., 2025; Gan et al., 2023; Tan et al., 2024).

The central problem addressed by this project is that many existing sign-language recognition systems are either not practical for everyday deployment or do not provide an adequate balance between vocabulary size, computational efficiency, and real-time usability. Earlier work in the field has explored sensor-based approaches using devices such as Leap Motion, Kinect cameras, depth sensors, and wearable armbands. These systems often achieve accurate motion tracking because they capture depth or muscle-related information more directly than a standard webcam. However, they also require additional hardware, controlled setup conditions, and higher deployment cost, making them less suitable for ordinary use outside research or demonstration environments (Tan et al., 2024). This hardware dependency limits accessibility, which is contrary to the broader aim of assistive technology.

Another major branch of sign-language research uses full RGB video and deep neural networks to learn directly from visual appearance. Such approaches can achieve impressive results when trained on large datasets and powerful computational infrastructure. However, they often rely on high-end GPUs, large memory budgets, and computationally heavy architectures that are difficult to run in real time on standard consumer laptops. In addition, raw video approaches frequently involve greater sensitivity to background complexity, lighting variations, and irrelevant visual detail (Camgoz et al., 2020; Duarte et al., 2021; Tan et al., 2024). As a result, while RGB-based systems may perform strongly in benchmark settings, they do not always translate cleanly into lightweight real-time applications intended for everyday users (Tan et al., 2024).

More recent research has increasingly favoured pose-based or keypoint-based pipelines. In these systems, video frames are first converted into structured landmarks representing the positions of the hands, fingers, face, and selected body joints. These compact coordinate representations are then modelled using temporal classifiers such as LSTMs, GRUs, Transformers, or graph-based networks (De Coster et al., 2023; Tan et al., 2024). This strategy offers several advantages. It drastically reduces the dimensionality of the input, removes much of the irrelevant background variation, and allows the model to focus on movement structure rather than raw image appearance (Holmes et al., 2024; Tan et al., 2024). For this reason, keypoint-based pipelines are especially attractive for real-time deployment on modest hardware (Kamble, 2025; Uddin et al., 2025). Even so, many systems in this category remain limited to small vocabularies such as alphabets, digits, or a narrow set of common gestures. Others omit speech output, thereby stopping at recognition rather than providing a usable translation interface (Alsharif et al., 2025; Kamble, 2025; Uddin et al., 2025).

Across the literature, a consistent gap emerges. There is a need for a lightweight, webcam-based, real-time American Sign Language system that supports a meaningful mid-sized vocabulary, runs efficiently on consumer hardware, and provides immediate translated output in text and speech form. Large-scale datasets such as WLASL enable research on isolated ASL sign recognition across many classes and signers, but isolated recognition alone is not yet the same as a practical application (Li et al., 2020). Continuous sign-language translation datasets such as RWTH-PHOENIX-Weather and How2Sign support richer sequence modelling, but they are often computationally demanding and not optimized for low-latency consumer deployment (Camgoz et al., 2020; Duarte et al., 2021). Meanwhile, many real-time prototype systems demonstrate feasibility only at small vocabulary scale (Alsharif et al., 2025; Kamble, 2025; Uddin et al., 2025). This project is motivated directly by that gap.

The present work therefore aims to build and evaluate a real-time ASL recognition and translation prototype that balances practicality and scale. The system accepts webcam input, extracts MediaPipe landmarks, normalizes and encodes them as temporal sequences, classifies the sign using a BiLSTM-based model trained on a 300-gloss subset of WLASL, and presents the result both visually and through text-to-speech. The project does not attempt full continuous sentence-level translation. Instead, it focuses on isolated sign recognition as a tractable and meaningful intermediate target. This scope is appropriate because isolated sign recognition remains a necessary building block for more advanced continuous systems, and because it allows the project to investigate whether a mid-sized vocabulary can be handled efficiently under real-time constraints.

The significance of the project is both practical and academic. Practically, the project explores a route toward low-cost communication assistance that can be demonstrated on ordinary laptop hardware. Academically, it contributes to the growing body of work on efficient sign-language recognition by examining how preprocessing parity, keypoint engineering, temporal modelling, and runtime stabilisation interact in a real deployment pipeline. Rather than treating model training and application deployment as separate concerns, the project addresses them as parts of a single system. This systems perspective is important because in real-time sign recognition, usability depends not only on dataset accuracy but also on motion gating, latency control, output stability, and the ability to manage near-miss predictions without excessive false positives (Gan et al., 2023; Kamble, 2025; Tan et al., 2024).

The project is also significant because it addresses an implementation gap that often appears in accessibility-focused machine learning work. Several real-time studies emphasize recognition on limited vocabularies, while other assistive prototypes address communication support under narrower or different task definitions (Alsharif et al., 2025; Badadhe et al., 2025; Kamble, 2025; Uddin et al., 2025). This project attempts to combine both. It implements a working user-facing pipeline while also operating at a vocabulary scale large enough to reveal real modelling difficulty. In doing so, it produces lessons that are more transferable than those from very small proof-of-concept systems. Even where the system remains imperfect, those imperfections are informative because they arise under meaningful project conditions rather than in a simplified classroom-only scenario.

The problem statement underlying the project may therefore be expressed clearly: existing approaches do not yet provide an accessible, lightweight, webcam-based ASL system that can recognize a substantial vocabulary and translate the results into speech in real time on ordinary consumer hardware. The project addresses this problem through a keypoint-based architecture informed by prior literature, implemented using practical engineering constraints, and evaluated through both offline and live testing.

The project objectives reflect this focus. The first objective is to build a real-time ASL recognition pipeline using standard hardware, with a regular webcam as the only input device and no specialized sensors. The second objective is to train a deep-learning model capable of recognizing 300 isolated ASL signs using the WLASL300 benchmark and a preprocessing strategy that improves robustness while remaining computationally efficient. The third objective is to deliver low-latency output in both text and speech, creating an end-to-end sign-to-speech interface rather than a classifier that only produces internal predictions.

These objectives are aligned with concrete success criteria. The system should be deployable on a single laptop and maintain smooth frame processing. It should apply keypoint normalization, hand-focused landmark filtering, and missing-keypoint handling while achieving meaningful recognition performance over 300 classes. It should also deliver live output with minimal flicker or delay, displaying predictions clearly and speaking stabilized signs through a text-to-speech module. Although perfect recognition is unrealistic for a 300-class isolated sign problem on student-level hardware, the project aims to demonstrate that useful real-time performance can still be obtained with appropriate design choices.

The development of this project was informed by a substantial literature background. Li et al. (2020) introduced WLASL as a large-scale benchmark for word-level ASL recognition and showed that large-vocabulary isolated sign recognition remains challenging even for strong models. Kamble (2025) demonstrated that MediaPipe plus LSTM can support real-time sign recognition on consumer hardware, though within a limited vocabulary. Camgoz et al. (2020) showed the power of temporal modelling and intermediate gloss representations for continuous sign translation, though in a computationally heavy setting not intended for low-latency consumer deployment. Duarte et al. (2021) provided a large multimodal ASL dataset and reinforced the importance of multimodal structure, while also highlighting the cost and scale of richer continuous-sign resources. Tan et al. (2024) synthesized broader trends in deep-learning-based sign-language processing and emphasized the growing role of keypoint-based pipelines for deployable systems. Uddin et al. (2025) and Alsharif et al. (2025) further validated the practicality of MediaPipe-based real-time recognition under small-vocabulary conditions. De Coster et al. (2023) emphasized the importance of normalization and robust sign embeddings, and Holmes et al. (2024) demonstrated that careful landmark selection can improve efficiency and accuracy. Gan et al. (2023) showed that even sign recognition and translation on edge devices can be approached using region-aware skeleton models, reinforcing the broader feasibility of efficient keypoint-based design.

Each of these studies influenced a specific aspect of the project. Li et al. (2020) justified the use of WLASL by showing that signer diversity and large gloss inventories are necessary if sign recognition research is to move beyond narrow, overcontrolled demonstrations. Kamble (2025), Uddin et al. (2025), and Alsharif et al. (2025) made clear that a webcam plus MediaPipe plus recurrent-sequence pipeline is not only theoretically sound but also practically implementable on accessible hardware. Camgoz et al. (2020) motivated the inclusion of explicit temporal modelling and reinforced the idea that sign interpretation should be treated as a sequence problem, even if the present project remains at the isolated-word level rather than full sentence translation. Duarte et al. (2021) helped frame the broader research landscape by showing what becomes possible with very large multimodal corpora, while also making visible the computational gap between such research settings and lightweight deployment. Tan et al. (2024) provided a useful synthesis that linked many of these trends together and highlighted the trade-off between accuracy-oriented heavy systems and latency-oriented pose-based systems. Holmes et al. (2024) and De Coster et al. (2023) strengthened the justification for targeted preprocessing, normalization, and discriminative landmark selection. Gan et al. (2023) further supported the idea that region-aware pose modelling can underpin real-time recognition and translation behaviour on constrained hardware.

The project’s significance can therefore be framed in two complementary ways. First, it has immediate applied significance because it explores a low-cost assistive communication pipeline that can operate with equipment already available to many users. Second, it has methodological significance because it investigates how a mid-scale vocabulary system can be made practical through careful trade-offs rather than by simply reducing the problem to a tiny gesture set. This distinction matters because several real-time systems in the recent literature focus on small alphabets, digits, or narrow gesture subsets rather than a broader word-level vocabulary (Alsharif et al., 2025; Kamble, 2025; Uddin et al., 2025). By targeting 300 classes, the present project places itself under more realistic lexical pressure and therefore produces findings that are more informative for future deployment-oriented research.

Taken together, these works justify the project’s architecture. A keypoint-based approach is appropriate for efficiency (Holmes et al., 2024; Tan et al., 2024). A temporal model is necessary because signs are dynamic rather than static gestures (Camgoz et al., 2020; Kamble, 2025; Uddin et al., 2025). Landmark normalization and feature selection are important for robustness and computational economy (De Coster et al., 2023; Holmes et al., 2024). A mid-sized dataset such as WLASL300 offers a meaningful compromise between realism and tractability (Li et al., 2020). And real-time deployment requires more than raw classification; it also requires smoothing, gating, and stable output logic (Gan et al., 2023; Kamble, 2025).

This report therefore treats the project as a full engineering and research effort rather than as a narrow model-training exercise. The following sections describe the methodology used to design and build the system, present the results obtained from implementation and testing, interpret those results in relation to the literature and project objectives, and conclude with recommendations for future work. Although live testing remains an important part of the evaluation, it is presented here as one component within a broader project narrative that includes architecture, preprocessing, modelling, offline evaluation, deployment, and usability considerations.

## CHAPTER 2 LITERATURE REVIEW

The literature reviewed for this project spans isolated sign recognition benchmarks, real-time webcam-based systems, continuous sign translation research, pose-based modelling studies, and deployment-oriented assistive prototypes. As the present project lies between benchmark-driven academic research and usable consumer-hardware deployment, the review is organized to move from foundational benchmark and sequence-modelling work toward studies that are closer to real-time implementation and assistive use. Each study is discussed on its own terms and then related briefly to the present project.

### 2.1 Word-Level Deep Sign Language Recognition from Video: A New Large-Scale Dataset and Methods Comparison

Li et al. (2020) argue that progress in sign language recognition had been constrained by the lack of sufficiently large and diverse datasets, and they address this problem by introducing the Word-Level American Sign Language dataset, commonly known as WLASL. Their work is important because it establishes a benchmark for isolated, word-level ASL recognition at a scale significantly larger than that of many earlier datasets. The dataset contains more than 2,000 glosses and over 100 signers, making it suitable for studying signer variability, class imbalance, and realistic recognition difficulty. The paper also compares appearance-based and pose-based methods, thereby framing an important methodological question for later work: whether full RGB representations or extracted keypoints are more suitable for scalable sign recognition.

The reported results show that recognition on WLASL remains challenging even for strong models, with baseline performance leaving substantial room for improvement. This makes the dataset especially valuable because it prevents the field from drawing overly optimistic conclusions from small, controlled vocabularies. WLASL therefore functions not only as a training resource but also as a difficulty benchmark that exposes the practical limits of sign recognition systems.

This study is directly aligned with the present project because WLASL300 is the project’s core dataset. The signer diversity, unconstrained recording conditions, and vocabulary scale of WLASL provide the foundation for training a model that aspires to be more than a narrow classroom demo. However, Li et al. (2020) do not present a lightweight sign-to-speech deployment system. Their study is primarily benchmark-driven and focused on dataset creation and method comparison. The present project builds on that benchmark by turning a subset of the dataset into a live webcam-based application with prediction stabilization and text-to-speech output.

### 2.2 SLRNet: A Real-Time LSTM-Based Sign Language Recognition System

Kamble (2025) presents SLRNet, a webcam-based sign language recognition system that combines MediaPipe Holistic with an LSTM-based recognition model. The work is notable because it explicitly targets real-time operation on consumer-grade hardware rather than only offline recognition experiments. Its pipeline extracts full-body, facial, and hand landmarks and feeds them into a lightweight recurrent model, with the aim of achieving usable inference speed and reasonable recognition accuracy. This places the study firmly in the category of deployment-oriented sign recognition research.

The reported performance, including validation accuracy around 86.7% and inference time on moderate hardware, supports the claim that landmark-based recurrent models can be practical for real-time sign recognition. However, the vocabulary size is limited, largely covering alphabet signs and a modest set of functional words. The system also emphasizes recognition rather than full translation support.

SLRNet is highly relevant to the present project because it validates the methodological core of MediaPipe plus LSTM-style temporal modelling for real-time use (Kamble, 2025). The present project differs in two important ways. First, it scales the recognition target to 300 isolated ASL signs, which makes the classification problem substantially harder. Second, it integrates text-to-speech output and a deployment pipeline that is explicitly designed around a WLASL300 benchmark workflow rather than a smaller webcam-focused vocabulary.

### 2.3 Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation

Camgoz et al. (2020) present a Transformer-based framework for continuous sign language recognition and translation. Their work is influential because it shows that recognition and translation can be modelled jointly rather than as two loosely connected stages. The study uses a sequence-to-sequence architecture and demonstrates that gloss-level intermediate representations can significantly improve downstream translation quality. It is particularly important in showing that temporal modelling for sign language is not merely helpful but central to capturing linguistic structure across sequences.

The reported BLEU improvements over earlier baselines indicate that attention-based architectures can model sign-language translation effectively when sufficient data and computational resources are available. However, the system is designed for research-scale continuous sign translation, not lightweight real-time deployment on standard laptops. It operates in a substantially different regime from isolated sign recognition and assumes the availability of a demanding dataset and a larger computational budget.

Camgoz et al. (2020) are relevant to the present project because they justify the use of explicit temporal sequence modelling and reinforce the idea that sign interpretation should not be treated as a static-image problem. However, the present project differs fundamentally in scope and deployment target. It focuses on isolated ASL signs rather than continuous German sign language translation, prioritizes low-latency inference over large-model translation quality, and uses a BiLSTM-based consumer-hardware pipeline rather than a heavy end-to-end Transformer translation system.

### 2.4 How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language

Duarte et al. (2021) argue that large-scale multimodal resources are essential for advancing automatic sign-language recognition and translation. They address this problem by introducing How2Sign, a large ASL dataset containing continuous signing videos, transcripts, gloss alignments, and multiple modalities including RGB, depth, and 2D or 3D pose information. The value of this work lies in its breadth. It provides researchers with a richly annotated corpus for studying continuous sign understanding, multilingual interaction, and multimodal modelling.

The dataset contains tens of thousands of aligned sentence-level instances and millions of extracted keypoints, making it one of the most ambitious resources for ASL research. However, that richness comes at a cost: data volume, annotation complexity, and computational demand are all much higher than those of isolated-sign datasets such as WLASL300. As a result, How2Sign is more suitable for large-scale continuous translation research than for lightweight real-time consumer deployment.

This work is relevant to the present project because it highlights the importance of pose information and multimodal structure even when the full multimodal stack is too expensive for a student-level real-time system. The present project differs in that it deliberately narrows the problem to isolated sign recognition and chooses a keypoint-based representation for efficiency. In other words, where How2Sign demonstrates what is possible with large multimodal corpora, the present project explores what remains possible when the goal is practical deployment on accessible hardware.

### 2.5 A Review of Deep Learning-Based Approaches to Sign Language Processing

Tan et al. (2024) provide a broad survey of deep-learning-based sign-language processing methods across recognition, translation, and generation. The study is valuable because it organizes the field by input modality, modelling architecture, and task type, thereby offering a conceptual map of how sign-language AI has evolved. It highlights the transition from handcrafted-feature pipelines to deep learning, the growing influence of attention and Transformers, and the trade-offs between appearance-based systems and keypoint-based approaches.

One of the most useful contributions of Tan et al. (2024) is the practical framing of these trade-offs. The survey emphasizes that raw RGB systems often perform strongly in data-rich settings but remain computationally heavy, whereas pose- or keypoint-based systems are more suitable for lightweight, real-time deployment. It also highlights robustness strategies such as keypoint normalization, augmentation, and temporal alignment, all of which are highly relevant to the design of real-time systems.

This review is closely aligned with the present project because it provides direct support for the project’s architecture. The reliance on MediaPipe keypoints, temporal modelling with a recurrent backbone, and a mid-scale benchmark such as WLASL300 all fit the design trends identified by Tan et al. (2024). The present project differs only in that it is not a survey but an implementation. It translates many of the survey’s field-level observations into an operational sign recognition and sign-to-speech system.

### 2.6 Real-Time Norwegian Sign Language Recognition Using MediaPipe and LSTM

Uddin et al. (2025) present a real-time sign recognition system for Norwegian Sign Language numbers using MediaPipe landmarks and an LSTM temporal model. The work is notable because it demonstrates that even without specialized hardware, a lightweight sign recognizer can achieve high accuracy on a standard webcam setup. By reducing the input to landmark coordinates and modelling them as a temporal sequence, the study reinforces the viability of efficient pose-based recognition pipelines.

The reported testing accuracy of 95% for the 0 to 10 number vocabulary shows that landmark-based temporal classification can be extremely effective for small sign sets. At the same time, the study acknowledges that overlapping hand configurations remain a challenge and that small-vocabulary success does not automatically scale to larger lexical tasks.

This work is methodologically similar to the present project because both use a MediaPipe-plus-LSTM design philosophy for real-time recognition (Uddin et al., 2025). However, the present project differs in three major ways: it targets ASL rather than Norwegian Sign Language, it handles a substantially larger 300-sign vocabulary, and it integrates text and speech output rather than stopping at recognition accuracy alone.

### 2.7 Real-Time American Sign Language Interpretation Using Deep Learning and Keypoint Tracking

Alsharif et al. (2025) present a real-time ASL interpretation system that uses MediaPipe keypoint tracking and a deep-learning classifier to convert gestures into text. The study is valuable because it demonstrates the practical viability of camera-based sign interpretation on standard hardware and emphasizes low-latency operation. Its use of landmark extraction aligns with broader research interest in replacing expensive sensing setups with accessible computer vision pipelines.

The reported performance is strong for the targeted task, and the paper shows that keypoint-based recognition can be accurate and responsive under realistic lighting and background conditions. However, the scope remains limited to alphabets and a relatively small gesture set. This means that while the system is useful as a demonstration of fast and practical interpretation, it does not address the difficulty of large-vocabulary word-level ASL recognition.

This study is therefore similar to the present project in hardware philosophy and use of MediaPipe-based landmarks, but different in scale and system ambition. The present project moves from alphabet-level or small-gesture recognition to a 300-sign WLASL-derived vocabulary and adds speech output, making it more aligned with mid-scale isolated sign translation than with alphabet-focused interpretation.

### 2.8 Towards the Extraction of Robust Sign Embeddings for Low-Resource Sign Language Recognition

De Coster et al. (2023) focus on extracting robust sign embeddings using pose estimation, normalization, and representation learning for low-resource sign language recognition. Their work is particularly important because it treats preprocessing not as a minor implementation detail but as a central part of model quality. The study investigates how keypoint extraction, normalization, and embedding learning can improve recognition and transfer across sign languages and low-resource settings.

The paper’s emphasis on normalization and handling noisy or incomplete pose data is directly relevant to any real-time sign-language application. By showing that preprocessing and embedding design can materially improve classifier robustness, the work strengthens the methodological legitimacy of investing heavily in the input pipeline rather than only in model depth.

This study is highly similar to the present project at the preprocessing level. Both rely on pose- or keypoint-based inputs and both treat normalization and missing-keypoint handling as essential rather than optional. The main difference is that De Coster et al. (2023) focus on low-resource embedding robustness and transferability, whereas the present project applies related ideas to a concrete 300-class ASL recognition and sign-to-speech deployment pipeline.

### 2.9 Towards Real-Time Sign Language Recognition and Translation on Edge Devices

Gan et al. (2023) introduce RTG-Net, a lightweight graph-based approach for sign recognition and translation on edge devices. The study is important because it moves beyond pure benchmark optimisation and explicitly addresses computational efficiency. By using key-region modelling and a lightweight architecture, the work demonstrates that low-resource deployment and meaningful sign-language processing need not be mutually exclusive.

The paper reports strong recognition and translation behaviour while reducing inference cost relative to heavier alternatives. This makes it particularly relevant to the present project, because it validates the broader idea that efficient sign-language AI can be built for practical hardware targets. At the same time, its architecture is more specialized and graph-centric than the BiLSTM-based approach used here.

Gan et al. (2023) are therefore similar to the present project in deployment philosophy, but different in implementation focus. RTG-Net is a more specialized research architecture for efficient edge processing, whereas the present project chooses a simpler and more accessible modelling stack centred on MediaPipe features, BiLSTM temporal modelling, and a full live webcam plus TTS interface.

### 2.10 The Key Points: Using Feature Importance to Identify Shortcomings in Sign Language Recognition Models

Holmes et al. (2024) investigate which keypoints matter most in sign-language recognition systems and which may be redundant or even harmful. This study is particularly useful because it does not simply propose another accuracy result. Instead, it analyzes existing pose-based models to determine where efficiency and recognition quality can be improved through more selective feature design.

Their findings suggest that removing less relevant landmarks can reduce dimensionality and sometimes improve performance. This is an important methodological insight for real-time systems, where every unnecessary input feature increases computational burden and may introduce noise. The work therefore helps connect explainability and deployment efficiency.

This paper is very similar to the present project at the feature-engineering level. The present project also prioritizes hand landmarks and selected upper-body cues rather than indiscriminately keeping every available point. The main difference is that Holmes et al. (2024) are conducting analytic feature-importance research, whereas the present project applies a similar philosophy in a concrete live recognition pipeline.

### 2.11 Modelling Sign Language with Encoder-Only Transformers and Human Pose Estimation Keypoint Data

Woods and Rana (2023a) present one of the closest benchmark studies to the present project in terms of vocabulary scope. Using human pose estimation keypoints derived from WLASL-alt and an encoder-only Transformer with a body-size normalization strategy, the study evaluates isolated-sign classification across 10, 50, 100, and 300 classes. Its results are especially notable at the 300-class setting, where it reports top-1, top-5, and top-10 accuracies of 71%, 90%, and 94%, respectively, while using models with fewer than 100k learnable parameters. This establishes an important benchmark for lightweight pose-based recognition at mid-scale vocabulary size.

The study is valuable because it demonstrates that high isolated-sign recognition accuracy can be achieved with compact models if the input representation and normalization scheme are effective. It also makes a strong case for careful benchmark reporting and repeated experimentation, rather than relying on single-run results.

This work is highly relevant to the present project because it establishes a credible accuracy target for pose-based 300-sign recognition. However, it remains an offline benchmark study. It uses OpenPose-derived keypoints from pre-recorded WLASL-alt videos rather than a live MediaPipe stream, does not include a webcam inference interface, and does not integrate text-to-speech or deployment latency considerations. The present project differs by taking a similar mid-scale recognition problem and embedding it in a full real-time application pipeline.

### 2.12 Constraints on Optimising Encoder-Only Transformers for Modelling Sign Language with Human Pose Estimation Keypoint Data

Woods and Rana (2023b) extend their earlier work by performing an ablation-style study on regularization and optimization choices for encoder-only Transformer models trained on WLASL-alt pose data. Rather than introducing a new end-to-end deployment system, the paper investigates how training decisions such as regularization affect sign-language classification performance. The study concludes that among the tested choices, `L2` parameter regularization has the clearest positive effect within the measured uncertainty.

This work is methodologically important because it shows that WLASL300-scale performance is sensitive not only to architecture but also to optimization constraints. In other words, meaningful accuracy changes can arise from training policy rather than from changing the entire model family.

For the present project, this study is useful as supporting evidence that regularization and training stability matter for mid-scale sign recognition. However, it is not a direct competing system, since it remains an offline OpenPose-based benchmark and does not address webcam integration, text-to-speech, or real-time inference constraints. The present project differs by prioritizing operational deployment rather than optimizer-focused ablation research.

### 2.13 Real-Time American Sign Language Recognition Using 3D Convolutional Neural Networks and LSTM: Architecture, Training, and Deployment

Key (2025) presents an arXiv preprint describing a hybrid 3D CNN plus LSTM system for real-time ASL recognition trained using WLASL, ASL-LEX, and a curated expert-annotated set. The study is notable because it combines strong spatiotemporal video modelling with deployment discussion, including AWS infrastructure and OAK-D edge-device compatibility. By using full RGB video and 3D convolutions, it targets a broader visual modelling regime than keypoint-only approaches.

The preprint reports class-wise F1-scores ranging from 0.71 to 0.99, which suggests that the architecture can model a range of sign classes effectively. It is also useful because it explicitly discusses deployment architecture rather than remaining purely offline. At the same time, its deployment assumptions depend on cloud or specialized-camera pathways that are different from a standard consumer-webcam pipeline.

This study is similar to the present project in that both are concerned with real-time word-level ASL recognition. However, it differs substantially in system philosophy. Key (2025) uses full RGB video input, 3D convolutions, cloud deployment support, and specialized OAK-D edge hardware. The present project instead uses a lighter MediaPipe keypoint stream, runs locally on consumer laptop hardware, and integrates text-to-speech output as part of a self-contained offline pipeline. The present project therefore trades some representational richness for greater practical accessibility and lower deployment complexity.

### 2.14 A Real-Time Webcam-Based System for Sign Language and Speech Translation

Kosna (2025) describes a ResearchGate-hosted preprint presenting a two-way translation system that combines sign recognition from webcam video with voice-to-sign translation through avatar output. The study is interesting because it is one of the few works that attempts bidirectional communication rather than only sign-to-text recognition. The manuscript reports a mixed-test-set recognition accuracy of 96.5% and also evaluates a small continuous-sign scenario using word error rate and sign error rate.

The paper’s main strength lies in its broad application ambition: it combines recognition, speech output, and sign synthesis in one conceptual framework. However, it relies on full RGB processing with a CNN-LSTM-attention architecture and a self-assembled multi-source dataset as an evaluation setup rather than a standard WLASL300 benchmark workflow. This makes direct benchmarking more difficult.

This work overlaps strongly with the present project in terms of end-goal, since both aim to bridge communication through a live system rather than a static benchmark. However, the present project differs by using a standardized WLASL-derived 300-sign benchmark, a keypoint-focused input pipeline, and a narrower but more controlled sign-to-speech scope. Kosna (2025) is therefore useful for deployment inspiration, but less suitable as a direct benchmark reference than dataset-standardized studies.

### 2.15 Real-Time Sign Language to Text Translation Using Deep Learning: A Comparative Study of LSTM and 3D CNN

Anturkar et al. (2025) compare 3D CNNs and LSTM networks for sign-language recognition on a 50-class ASL dataset comprising 1,200 signs. The study is useful because it explicitly quantifies the trade-off between accuracy and computational efficiency. According to the reported results, 3D CNNs achieve higher recognition accuracy, while LSTMs require much less processing time per frame. This makes the paper relevant to architecture selection for real-time systems.

The value of the paper lies less in its dataset scale and more in its comparative insight. It provides evidence that a lower-cost temporal model can remain attractive when deployment efficiency matters more than maximizing recognition accuracy at any cost.

This study supports the present project’s choice of a recurrent temporal model over a heavier 3D CNN stack. The systems differ in several respects: Anturkar et al. (2025) use a smaller custom vocabulary, do not present a full webcam plus TTS deployment system, and focus on comparative benchmarking rather than operational translation. The present project applies the efficiency argument at a larger vocabulary scale and within a full live application.

### 2.16 MIPA-ResGCN: A Multi-Input Part Attention Enhanced Residual Graph Convolutional Framework for Sign Language Recognition

Naz et al. (2023) propose MIPA-ResGCN, a multi-input graph-convolutional framework that combines pose extraction, handcrafted joint or bone features, and part attention to improve sign recognition. The paper is significant because it reports strong results on multiple datasets, including WLASL-300 where it achieves 72.90% top-1 accuracy. It also emphasizes reduced computational complexity relative to prior graph-based alternatives, making it an important point of comparison for efficient isolated-sign recognition.

The model is analytically rich and contributes a strong research result for pose-based sign recognition. Its use of attention over body parts and graph structure reflects the broader trend toward more structured spatiotemporal modelling beyond standard recurrent networks.

Naz et al. (2023) are highly relevant to the present project because they provide a realistic benchmark range for WLASL-300 using a pose-based method. However, the study remains an offline research model using pre-extracted pose information rather than a live MediaPipe stream, and it does not provide a webcam interface or TTS output. The present project differs by aiming for a simpler, more deployable architecture even if that means accepting a lower theoretical ceiling than a specialized graph model.

### 2.17 Ensemble Transformer-Based Word-Level Sign Language Recognition with Multi-Modal Input Fusion

Alkhoraif et al. (2025) present a two-stream ensemble model that fuses appearance and pose information using Swin-Transformer backbones. The paper reports state-of-the-art top-1 accuracy of 93.51% on WLASL, showing the power of multimodal Transformer-based modelling when the primary objective is recognition accuracy. Its strong performance across multiple datasets makes it an important reference point in the broader sign-recognition landscape.

The study’s contribution is substantial from a benchmark perspective. It shows that combining appearance and pose streams can produce very strong recognition performance, and it illustrates the continuing accuracy gains achievable through multimodal deep architectures.

This work is relevant to the present project as an upper-bound comparison, but it is not directly aligned with the project’s deployment goals. Running two Swin-Transformer streams is far more computationally expensive than a keypoint-based BiLSTM pipeline intended for low-latency live use on consumer hardware. The present project therefore differs by consciously trading benchmark-level accuracy for accessibility, simplicity, and real-time deployability.

### 2.18 Real-Time American Sign Language to Speech Conversion Using CNN and Computer Vision

Badadhe et al. (2025) propose a real-time ASL-to-speech system that uses a CNN classifier together with OpenCV, MediaPipe, and the `pyttsx3` speech engine. The system is aimed at practical sign-to-speech conversion and reports strong performance for static gesture recognition. Its key contribution lies in showing that a lightweight speech-output pipeline can be created with accessible tools and without reliance on remote infrastructure.

The reported accuracy range of 96% to 99% is impressive for the task the system addresses. However, the task itself is fundamentally narrower than dynamic word-level sign recognition, since it is based on static gesture recognition rather than temporal word-sign modelling.

This study is similar to the present project in one important respect: both use a speech-output layer to turn visual recognition into a communicative aid. However, the recognition backbone is fundamentally different. Badadhe et al. (2025) classify static gestures, whereas the present project is designed for dynamic isolated signs whose meaning unfolds across a motion sequence. The present project therefore extends beyond static recognition by modelling temporal sign trajectories through a BiLSTM-based pipeline.

### 2.19 SignSpeak – Sign Language Translation System for Hearing Impaired

Kalaiselvi et al. (2025) present SignSpeak, a real-time speech-to-sign translation system that uses Whisper automatic speech recognition, BERT-based processing, and gloss-video retrieval or stitching to generate sign-language output. The work is useful because it addresses an important communication problem and demonstrates a practical pipeline for translating spoken language into sign-language video presentation.

Its contribution is directionally different from most sign-recognition studies. Rather than classifying live sign input, it takes audio input, transcribes and reorders it, and retrieves sign gloss clips to produce a sign-language response. This makes it more relevant to sign synthesis and translation delivery than to webcam-based sign recognition.

SignSpeak is therefore only partially similar to the present project. Both projects aim to reduce communication barriers, but they solve opposite communication directions. The present project translates live signs into text and speech, whereas Kalaiselvi et al. (2025) translate speech into sign-video output. For that reason, SignSpeak is better suited to be discussed in the literature review as a contrastive assistive system than as a direct methodological source for other sections of the present report.

## CHAPTER 3 RESEARCH METHODOLOGY

This project adopted a design-and-evaluate research methodology grounded in system development, literature-informed decision making, empirical model training, and practical deployment testing. The aim was not only to train a sign-language classifier, but to construct an end-to-end prototype that could function in real time under realistic hardware constraints. The methodology was therefore modular and iterative, combining dataset preparation, feature engineering, model design, runtime implementation, and evaluation.

At the highest level, the project architecture consists of four interacting layers: video capture, landmark extraction and preprocessing, temporal sequence modelling, and output generation. The first layer acquires live webcam frames. The second layer converts those frames into a structured representation consisting primarily of hand landmarks, with optional upper-body pose and compact face information for disambiguation. The third layer models temporal dynamics across a rolling sequence of frames using a BiLSTM-based neural network with attention. The fourth layer stabilizes predictions, displays the output on screen, and optionally converts the recognized sign into speech. This modular design made it possible to improve one layer without discarding the rest of the pipeline.

The dataset choice was a foundational methodological decision. The project uses the WLASL300 subset derived from the larger WLASL dataset introduced by Li et al. (2020). WLASL is significant because it provides isolated ASL signs collected from many signers, thereby introducing signer diversity, variation in execution, and more realistic class structure than smaller demonstration datasets. A 300-gloss subset was chosen because it is large enough to make the recognition problem meaningful, yet still feasible for training and experimentation within student-level computational limits. This choice reflects an explicit trade-off between ambition and practicality. A very small vocabulary would not sufficiently address the real-world gap identified in the problem statement, while a much larger vocabulary or a full continuous translation corpus would likely exceed the project’s hardware and time constraints.

The data preparation pipeline was designed to preserve consistency between offline training and live deployment. Video clips from the dataset were processed into feature sequences using shared extraction and normalization logic. This emphasis on parity was influenced directly by the literature on robust embeddings and normalization, especially the work of De Coster et al. (2023), as well as by the general engineering principle that deployment failures often arise when training and inference preprocess input differently. In the active WLASL300 path, both training and runtime call the same shared feature-engineering code rather than maintaining two separate implementations. This matters because it means any remaining errors can be interpreted more confidently as modelling, calibration, or live-control problems rather than as preprocessing mismatch.

The saved parity artefact in `reports/preprocessing_parity.json` makes this concrete. It records `wlasl_runtime_matches_training = true` and `max_abs_diff_wlasl_runtime_vs_training = 0.0`, which means the active WLASL300 runtime reconstructs the same feature values used during training. The same artefact also records `max_abs_diff_legacy_main_vs_training = 1.2382903099060059`, showing why the older legacy path cannot be treated as parity-equivalent. This exact agreement on the active path matters because even small mismatches in feature order, normalization center, or gap handling can degrade deployment performance.

Landmark extraction was based primarily on MediaPipe because it offers real-time performance, broad accessibility, and a clean interface for obtaining hand and body coordinates from ordinary webcam frames (Alsharif et al., 2025; Kamble, 2025; Uddin et al., 2025). The project’s base representation uses hand landmarks from the left and right hands, giving `42 x 3` coordinate values per frame or 126 flattened features. The extended training path also supports selected pose and face features, especially for signs where hand information alone may be insufficient. This design choice was informed by multiple sources in the literature. Kamble (2025), Uddin et al. (2025), and Alsharif et al. (2025) all support the feasibility of MediaPipe-based recognition on standard hardware. Tan et al. (2024) identify pose-based pipelines as especially relevant for real-time deployable systems. Holmes et al. (2024) and De Coster et al. (2023) further motivate selective use of meaningful landmarks and careful preprocessing.

Several preprocessing operations were introduced to improve robustness. First, keypoint normalization was applied so that coordinate sequences would be less sensitive to signer size, camera distance, and absolute frame position. This was necessary because raw landmark coordinates alone can encode irrelevant variation unrelated to the sign identity (De Coster et al., 2023; Tan et al., 2024). Second, missing-keypoint handling or imputation was included to mitigate the effect of occasional detection drops by MediaPipe (De Coster et al., 2023). Third, feature selection was used to keep the representation compact and focused on the most relevant sign information (Holmes et al., 2024). Fourth, temporal sequence length was standardized by constructing a fixed-length rolling window for both training and live inference. These preprocessing steps were not incidental implementation details; they were core methodological responses to the limitations identified in the literature and to the realities of webcam-based recognition.

For the active face-aware path, each frame is encoded as a `180`-dimensional vector rather than only the `126` hand values. The first `126` values encode both hands. The next `21` values encode seven selected pose joints `(0, 11, 12, 13, 14, 15, 16)`, and the final `33` values encode eleven compact face landmarks `(10, 151, 168, 1, 2, 13, 14, 17, 152, 33, 263)`. This compact upper-body and face slice preserves body-relative and face-relative anchor information without requiring a full dense pose or face representation. It became especially useful for signs whose distinction depends on placement relative to the head or torso, such as `MOTHER` versus `FATHER`.

The active deployed checkpoint uses a fixed sequence length of `30` frames, so each model input has shape `30 x 180`. Short interior gaps are linearly interpolated feature by feature, while longer weak segments are not simply treated as valid motion. After this cleanup, variable-length clips are resampled onto a fixed temporal grid. In this report, a rolling window refers to the runtime behaviour of continuously keeping only the most recent `30` processed frames in memory so that the model always receives a temporally consistent input length.

The feature representation was constructed as a temporal sequence of normalized frame-wise landmark vectors. This sequence representation is compact enough for efficient learning and inference, yet expressive enough to model changes in hand shape, position, and motion over time. Unlike full RGB input, which contains textures, lighting variation, and background information, the landmark representation attempts to isolate the sign dynamics that matter most to recognition. This makes it especially suitable for a latency-conscious system.

The project also benefited from the fact that a structured keypoint representation supports explicit reasoning about feature quality. When raw RGB systems fail, it is often difficult to determine whether the problem is due to appearance clutter, camera noise, training bias, or an internal representational blind spot. By contrast, a landmark pipeline allows the developer to inspect whether hands are missing, whether pose geometry is skewed, whether left-right relations are visible, or whether the motion trajectory is underspecified. This level of visibility proved valuable during the project because it made both offline debugging and live calibration more concrete. The result is a system that may not solve every class perfectly, but whose failure cases are easier to understand and improve.

The primary sequence model used in the project is a Bidirectional Long Short-Term Memory network with attention. The choice of a BiLSTM rather than a large Transformer was deliberate. LSTMs remain well suited to moderate-length temporal sequences and are computationally lighter than many Transformer-based alternatives, especially in student-level implementation contexts (Kamble, 2025; Uddin et al., 2025). The literature provides strong support for temporal modelling in sign-language tasks. Kamble (2025) and Uddin et al. (2025) demonstrate that LSTM-based pipelines can be effective and efficient in real-time settings. Camgoz et al. (2020), although focused on a different and more computationally demanding task, reinforce the broader importance of temporal modelling for sign interpretation. The use of attention within the BiLSTM helps the model focus more strongly on the most informative frames within the sequence rather than treating all frames as equally important.

The final production-oriented configuration used a two-layer BiLSTM with `hidden_dim = 512`, `dropout = 0.5`, and `input_dim = 180`. Here, `input_dim = 180` means each frame contributes `180` scalar features to the model: hand, pose, and compact face values combined. `hidden_dim = 512` means each LSTM direction learns a relatively large internal state of `512` values for representing temporal context; this gives the model enough capacity to encode direction, rhythm, and body-relative structure across the sign sequence. `dropout = 0.5` means that during training, half of the intermediate units in the dropout-enabled layers are randomly ignored on each update step. This acts as regularization: it reduces over-reliance on any single internal activation pattern and helps the model generalize better to new signers and noisier webcam input.

A BiLSTM processes the sequence in both forward and backward directions, so each time step is informed by what came before and what came after. This is useful because a partially observed sign can remain ambiguous until the direction, destination, or completion of the movement becomes clear. The attention block then performs learned temporal weighting over the encoded sequence. In practical terms, it gives greater importance to the frames containing the most discriminative sub-motion and lower importance to transition frames, hesitation frames, or low-information pauses. The attention used here is therefore a learned pooling mechanism over time, not a full Transformer self-attention stack.

The training procedure incorporated several measures to improve generalization across the 300 classes. These included class-balanced sampling, augmentation, weighted loss, focal-style loss emphasis, and pose or face feature fusion during different experiment stages. These options were explored because large-vocabulary sign datasets typically contain class imbalance, inter-signer variability, and difficult near-neighbour confusions. Data augmentation was used to expose the model to moderate temporal and spatial variation. Class-balanced sampling helped prevent the model from overly favouring classes with more plentiful training examples. Weighted or focal-like losses were relevant for pushing the model to learn harder classes rather than optimizing mostly for already easy ones. Hyperparameters such as sequence length, batch size, and training duration were selected with the dual goals of model quality and deployment practicality in mind.

The final hardened face-aware run used the MediaPipe landmark cache as input, `50` epochs, batch size `32`, learning rate `1e-3`, `3` warmup epochs, a `ReduceLROnPlateau` scheduler, and a minimum learning-rate floor of `1e-5`. The run executed on `cuda` with mixed precision enabled and `num_workers = 0`. The dataset split sizes were `3,549` training samples, `900` validation samples, and `668` held-out test samples.

These training terms describe how optimization was controlled. `50` epochs means the model saw the training set `50` times in full. A batch size of `32` means each gradient update was computed from `32` sequences at a time. A learning rate of `1e-3` means the optimizer began with update steps of `0.001` in scale. The `3` warmup epochs mean the learning rate was ramped up gradually at the beginning rather than starting at full strength immediately, which helps avoid unstable early updates. `ReduceLROnPlateau` means the learning rate was automatically reduced when validation improvement stalled, helping the optimizer move from coarse exploration to finer adjustment. The learning-rate floor of `1e-5` prevented the step size from shrinking indefinitely. Running on `cuda` means training used the GPU. Mixed precision means some computations used lower-precision arithmetic where safe, which speeds up training and reduces memory use. `num_workers = 0` means data loading stayed in the main process, a conservative choice that avoids multiprocessing issues at the cost of some loading speed.

In this training setup, class-balanced sampling means the training loader did not draw every class with the same raw dataset frequency. Instead, a `WeightedRandomSampler` gave rarer classes a higher probability of appearing in a batch so that common classes would not dominate the optimization signal. Here, optimization signal means the summed evidence the model receives about how to update its weights during training. If frequent classes appear too often, the model learns disproportionately from them and may neglect rarer signs. Weighted loss and focal-style loss were explored as alternative ways to emphasize difficult classes, but they were not retained in the final selected checkpoint. Weighted loss means errors on selected classes count more strongly in the loss function. Focal-style loss means easy already-correct predictions are down-weighted so the model concentrates more on hard or misclassified examples.

Data augmentation was also stronger than in earlier runs. Coordinate jitter means small random perturbations were added to landmark coordinates so the model would not overfit to exact point locations. Smaller companion jitter for pose and face channels applied the same idea more conservatively to upper-body and face anchors. Finger-bone scaling in the range `0.9-1.1` means finger segment lengths were randomly compressed or expanded by up to about `10%` to simulate natural articulation differences. Per-frame full-hand occlusion means the entire hand feature block could be zeroed for individual frames, imitating temporary detector failure. One-hand dropout means only one hand was removed, forcing the model to cope with partial visibility. Temporal frame dropout means some frames inside the sequence were removed, imitating missed detections or unstable frame cadence. Random frame skipping before final resampling means the source clip was sampled with slight temporal irregularity before being mapped back onto the fixed `30`-frame grid. Together, these operations were introduced specifically to simulate occlusion, tracking flicker, hand loss, and unstable webcam cadence.

The project also followed an experiment-oriented methodology rather than a single fixed training path. Experiment launchers, alternative training configurations, confusion analysis utilities, and evaluation outputs were used throughout development. This is methodologically important because it means the model used in deployment was the outcome of iterative experimentation rather than an arbitrary first configuration. Different variants explored hand-only, pose-enhanced, and pose-plus-face representations, as well as balancing and augmentation choices. Such exploration aligns with the literature’s broader observation that sign recognition performance is highly sensitive to representation and modelling choices (Holmes et al., 2024; Tan et al., 2024; Woods & Rana, 2023b).

This experimentation strategy was especially valuable because the problem space contains multiple competing priorities. Increasing model complexity may improve class discrimination but harm real-time latency. Adding more landmarks may improve contextual understanding but also increase noise and dimensionality. Aggressive smoothing may stabilize output but suppress short valid motions. Lowering thresholds may improve recall but also increase false positives. The project therefore had to treat system development as a balancing exercise rather than a single optimization problem. The presence of multiple experiment paths and reports reflects that balancing process and shows that the final system configuration emerged from comparative development rather than guesswork.

The experiment launcher `scripts/run_experiment_matrix.py` formalized several of these comparisons. Its predefined matrix includes a balanced-sampling plus augmentation baseline, a pose-enhanced run, a pose-plus-face run, a longer `36`-frame run, and a focal-loss run. The broader changelog records how related ideas were tested in practice. The hand-only high-capacity run `asl_model_300_bilstm512_plateau_v1` reached `60.67%` validation Top-1, `57.50%` test Top-1, and `83.57%` test Top-5. Adding pose features in `asl_model_300_bilstm512_pose_v1` improved this to `68.00%` validation Top-1, `63.03%` test Top-1, and `84.59%` test Top-5. The targeted longer-window pose run `asl_model_300_bilstm512_pose_seq40_targeted_v1` reached `68.10%` validation Top-1, `61.82%` test Top-1, and `85.80%` test Top-5. The earlier face-aware balanced-augmentation run `asl_model_300_pose_face_balaug_v1` achieved `67.46%` validation Top-1 and `60.03%` test Top-1, while the final hardened face-aware run improved these figures to `72.31%` validation Top-1, `65.01%` test Top-1, and `87.59%` test Top-5.

These trade-offs can be stated more explicitly. In this report, noise refers to variation in the input that does not help distinguish the sign, such as jitter, brief tracking failures, or inconsistent pose estimates. Dimensionality refers to the number of scalar input features; increasing the feature width can provide more context, but it also increases the amount of information the model must learn to use correctly. Smoothing refers to temporal filtering or vote aggregation intended to reduce rapid output changes. Recall refers to how often the system successfully detects the correct sign when that sign is actually present. Lowering a confidence threshold can improve recall by allowing more borderline correct predictions through, but it can also increase false positives by allowing more wrong predictions to surface.

Real-time deployment required additional methodology beyond training. A trained classifier alone is not sufficient to produce stable live output. The project therefore introduced a runtime inference pipeline built around a rolling temporal sequence buffer, top-k prediction tracking, stabilization logic, and text-to-speech. A rolling temporal sequence buffer means the system continuously stores only the most recent `30` processed frames and discards older ones as new frames arrive. This creates a moving window of current signing context instead of treating every frame independently. As webcam frames are processed, normalized features are added to this temporal buffer until a full window is available. The classifier then predicts a distribution over the 300 glosses. However, rather than emitting the top class immediately on every frame, the runtime applies multiple control mechanisms. These include confidence thresholds, adaptive fallback behaviour, per-sign overrides, motion requirements for signs prone to false activation, confusion-pair suppression, and peak detection for short but strong predictions. This deployment methodology reflects the reality that real-time human-facing systems must optimize not only accuracy but also stability and trustworthiness.

Top-k prediction tracking means the runtime preserves not only the single best class but also the next most likely candidates, with the UI currently displaying the Top-5 list. Stabilization logic means that these predictions are accumulated across a `10`-prediction history window rather than trusted frame by frame. The current WLASL300 defaults use a base confidence squelch of `0.65`, an adaptive lower acceptance floor of `0.45`, a runner-up margin of `0.12`, a `10`-prediction stabilization window, and a minimum vote count of `6`. In practical terms, `0.65` means a sign normally needs at least `65%` confidence before it can contribute to a stable output. The lower fallback floor of `0.45` means some signs can still be considered from `45%` upward, but only under stricter conditions. The runner-up margin of `0.12` means the best sign must lead the next-best candidate by at least `12` percentage points before the adaptive fallback will trust it. The `10`-prediction window means the system looks back across the latest ten accepted prediction steps, and the minimum vote count of `6` means at least six of those ten must agree before the sign is committed. If the best sign clears the normal threshold, it enters the vote history immediately. If it falls below the global threshold but still remains clearly ahead of the runner-up, the adaptive fallback logic can still accept it for selected weaker live signs.

Motion requirements are applied only to selected signs that should not fire while the hands are almost static. Confusion-pair suppression checks whether a known rival sign remains too close in confidence and, if so, holds the output instead of committing too early. Peak detection handles another failure mode: some signs hit the correct class only briefly at their most expressive frame and then decay into a noisier end pose before the normal vote window fills. For signs such as `ARRIVE`, `CATCH`, `HOPE`, `JACKET`, and `LAW`, a separate short peak history can preserve that correct peak rather than letting it disappear immediately.

The text-to-speech component forms the project’s translation layer. While the system does not perform full sentence translation, it translates recognized isolated signs into spoken output, thereby converting classification into a more directly useful assistive interaction. The project configuration uses `pyttsx3` as the active offline TTS backend, and on Windows the implementation prefers the native SAPI speech path when available for more reliable repeated utterances across a session. Integrating TTS into the methodology mattered because the project’s value lies partly in whether it can communicate results beyond an on-screen label. A sign recognizer that only prints class names is informative for technical evaluation; a sign recognizer that speaks the output begins to address the communication problem stated at the start of the project.

The user interface and runtime feedback loop were also part of the methodology. The live application was designed to display ongoing predictions, stabilized outputs, and camera feed or landmarks. This visual feedback allowed the tester to diagnose system behaviour, observe confusion patterns, and refine sign execution. It also made the system more transparent during demonstrations. In a practical assistive setting, transparency matters because users need to understand whether the system is uncertain, whether a sign has been recognized, and whether speech output has been triggered intentionally rather than accidentally.

The interface therefore functioned as a diagnostic tool as well as a presentation layer. Visually, the system shows the live camera frame as the main canvas. Over this frame, a semi-transparent prediction panel is drawn in the upper-left region. The panel includes the current stabilized sign label, the confidence assigned to that stabilized output, and a ranked Top-5 guess list. Each Top-5 row includes the sign name, a horizontal confidence bar, and a percentage value, with the highest-ranked guess emphasized in green and lower-ranked competitors shown in orange. This layout is important because it lets the tester see not only the final answer but also the nearby rivals competing for output.

Additional interface elements report system state. An FPS counter is shown near the upper-right corner so the user can verify that the live loop is maintaining interactive speed. A status line is drawn near the bottom of the frame to describe what the pipeline is currently doing. A readiness indicator with a colored circular light appears near the lower-right corner: green when the model is ready to infer on the expected input width and red when it is not yet ready. When the model is using the face-aware path, the label can also show the expected feature width, such as `180D`. If landmark overlay is toggled on, hand and pose points are drawn directly on the video: right-hand points in green, left-hand points in blue, and pose points in yellow. This makes it possible to inspect whether recognition problems come from the model itself or from upstream landmark extraction failure.

The textual statuses also encode different runtime phases. `Buffering 17/30` means the system has not yet accumulated enough frames to form a full `30`-frame input window. `Holding context 4/10` means the hands briefly disappeared, but the `10`-frame grace period is still preserving the last valid temporal context. `Waiting for hands` means the grace period has expired and the sequence buffer is no longer being carried forward. `Stable sign: ...` means the sign has already met stabilization requirements and is eligible for display and speech. `Listening for a stable sign...` means hands are visible but the accumulated evidence is still too weak or too inconsistent to commit. These interface components made the live system interpretable during testing instead of behaving like a black box.

Evaluation was conducted through both offline and live measures. Offline evaluation used held-out dataset splits and generated classification reports, weak-class summaries, confusion analyses, confusion matrices, and preprocessing parity checks. These metrics provided a formal measure of model performance under controlled sample-based conditions and quantified not only overall accuracy but also error structure. Live evaluation, by contrast, examined system behaviour in direct webcam use. This included whether the system could recognize signs in real time, whether predictions were stable, whether text-to-speech fired at appropriate moments, and which classes required sign-specific runtime correction or user adaptation. The live-testing process therefore complemented rather than replaced offline evaluation.

Offline evaluation in this project specifically used the held-out test split of `668` samples. Overall sample-level accuracy was `0.5195`, meaning that the model assigned the correct Top-1 label to about `51.95%` of the held-out sign sequences. The macro-average precision was `0.5321`, macro-average recall was `0.5276`, and macro-average F1-score was `0.4954`. These macro scores give every class equal weight, so they answer the question: how well does the system perform if rare and common signs are treated as equally important? The weighted-average precision was `0.5354`, weighted-average recall matched the accuracy at `0.5195`, and the weighted-average F1-score was `0.4931`. These weighted scores account for class support, so classes with more test examples influence the final figure more strongly. At class level, precision means how often a predicted label is correct when the model outputs it, recall means how often the model successfully recovers the true class when that sign actually appears, and F1-score summarizes the balance between the two in a single number.

The per-class results were highly uneven, which is typical for a 300-class isolated-sign problem. Stronger classes included `HELP`, `NO`, and `TABLE`, each with precision `1.000`, recall `1.000`, and F1-score `1.000` over support `3`; this means that all three held-out examples for each of those glosses were recovered correctly, and the model did not assign those labels incorrectly elsewhere in the evaluated subset. `HOW`, `MANY`, and `ORANGE` each reached recall `1.000` with F1-score `0.857` over support `3`, which means the true examples were all recovered but a small number of extra incorrect predictions lowered precision. `BOOK`, with precision `0.600`, recall `0.750`, and F1-score `0.667` over support `4`, illustrates a more mixed class: most true examples were found, but not all, and some `BOOK` predictions were wrong. At the other end, weak classes with support of at least `3` included `BEFORE`, `WHO`, `FINE`, `THIN`, `CORN`, `PINK`, `BIRTHDAY`, `BACKPACK`, `BAR`, `CHECK`, and `FAR`, all of which had recall `0.000` on the held-out split. Here, recall `0.000` means that none of the test examples for those glosses were correctly recovered. The most frequent confusion pathways were `CHAIR -> TRAIN`, `DEAF -> GOVERNMENT`, `KISS -> MORE`, `DOCTOR -> LEARN`, `BUT -> DIFFERENT`, `SON -> DAUGHTER`, `BAR -> YESTERDAY`, `CHECK -> LETTER`, `MOVIE -> BUSINESS`, and `SALT -> TRAIN`, each appearing twice in the summarized confusion list.

Qualitative evaluation also played an important role. In assistive interface projects, a system may appear numerically acceptable while still behaving awkwardly or confusingly in practice. For that reason, the project paid attention to factors such as whether predictions flickered excessively, whether the system recognized signs only under impractically narrow hand placements, whether speech output felt appropriately timed, and whether a user could understand why the recognizer had succeeded or failed. These observations were not treated as secondary anecdotes. They were part of the evaluation because they spoke directly to the project’s purpose as a communicative tool rather than a benchmark-only model.

The performance targets guiding this evaluation were practical rather than purely theoretical. The project aimed for usable real-time performance on standard hardware, low-latency frame processing, meaningful Top-1 and strong Top-5 recognition behaviour on WLASL300, and stable live output without excessive flickering or idle false positives. The project also sought robustness under occasional landmark dropouts or imperfect webcam conditions. These targets were informed both by the literature and by the needs of real-world interaction.

These targets were tested against specific live behaviours rather than only abstract goals. For example, the system initially needed to suppress idle false positives, reduce flicker, and preserve context during brief hand loss. The addition of explicit idle-state clearing, a `10`-frame grace period, and a `10`-prediction vote buffer with a `6`-vote requirement came directly from those observed live failures. A concrete example was the false appearance of `LATE` when the user was not signing at all, which showed that the runtime needed stronger idle handling instead of relying only on raw model output.

The live-testing methodology itself was structured and exhaustive. A 300-word test vocabulary corresponding to the deployed sign set was used to assess operational recognition. For each sign, observations were recorded about whether the sign was recognized immediately, recognized after refinement, present only in the top five, absent from the top five, or still pending due to unsatisfactory runtime behaviour. Notes were also recorded about hand orientation, signing speed, visibility, or confusion partners. Although this evaluation involved iterative adaptation rather than a blind randomized protocol, it served an important methodological purpose: it exposed the practical boundary between model capability and deployment usability.

The methodology also accounted for validity considerations. Construct validity is supported because the system was evaluated directly against its intended function: recognizing isolated signs and producing usable output in real time. Ecological validity is stronger than in purely offline evaluation because the system was tested in its deployment form using a webcam and live human input. Internal validity is somewhat limited by the role of iterative adaptation in live testing, but that adaptation is itself relevant to deployment behaviour. Reliability is supported by the size of the tested vocabulary and by the persistence of repeated confusion patterns across signs.

These validity terms can be restated more plainly. Construct validity concerns whether the evaluation actually measures the task the system was built to perform; here, that task was isolated sign recognition with visible and spoken output. Ecological validity concerns whether the testing condition resembles real use; the webcam-based live trials satisfy this better than a benchmark-only experiment. Internal validity concerns how confidently changes in behaviour can be attributed to system changes rather than uncontrolled factors; this is weaker in live trials because the tester also adapted articulation, but that adaptation is itself part of the real deployment problem. Reliability concerns whether the same kinds of successes and failures recur consistently; the repetition of weak classes such as `BEFORE`, `FINE`, and `KISS`, and confusion pairs such as `KISS/MORE` and `BUT/DIFFERENT`, across both held-out evaluation and live testing supports that reliability.

From a project-management perspective, the methodology was iterative and evidence-driven. Observed errors led to confusion analysis, targeted runtime rules, and adjusted execution strategies. Training and deployment were therefore linked through a feedback loop. This approach is consistent with the needs of a practical machine learning system, where a usable result often emerges not from a single training run but from repeated cycles of modelling, testing, and correction.

This feedback loop can be stated concretely. When offline confusion analysis showed anchor-location problems such as `MOTHER -> FATHER`, the representation was expanded beyond hands alone. First, selected pose joints were added so that the model could see relative placement around the shoulders, elbows, and torso. Then compact face anchors were added so signs made near the head could be interpreted relative to a stable facial reference rather than only to the hands themselves. When live testing showed idle false positives such as `LATE`, idle-state clearing and low-motion handling were introduced so the system would stop carrying stale predictions after signing had already ended.

When correct signs peaked only briefly and then decayed into a wrong end pose, peak-sign stabilization was added. This created a second, shorter memory for signs that briefly crossed a stronger threshold, allowing short-lived correct spikes such as `ARRIVE`, `CATCH`, `HOPE`, `JACKET`, and `LAW` to remain available instead of vanishing before the normal vote history filled. When signs repeatedly appeared inside the Top-5 list but failed to finalize, per-sign confidence overrides and confusion-pair suppressors were introduced. A confidence override means a specific sign can be accepted at a slightly lower threshold than the global default when repeated live testing shows that it is often correct but systematically under-confident. A confusion-pair suppressor means the system deliberately withholds output if a known rival class remains too close in score, reducing premature commitments in pairs such as `KISS/MORE`, `BUT/DIFFERENT`, or `DOCTOR/LEARN`.

When brief hand loss destroyed otherwise correct long signs, the `10`-frame grace period was added. Instead of clearing the entire sequence immediately when hands disappeared for a moment, the runtime now preserves the last valid temporal context for up to ten frames. If the hands reappear within that period, inference resumes from the preserved context rather than restarting from an empty buffer. This was important because sign execution in live webcam use often includes short detector dropouts that should not be treated as the end of the sign.

In summary, the research methodology combined literature-guided design, dataset-driven model training, feature engineering, real-time systems development, offline quantitative evaluation, and live operational testing. This methodology was appropriate for the project’s purpose because the project aimed not merely to achieve benchmark accuracy, but to demonstrate a functioning, lightweight, mid-vocabulary ASL sign-to-speech system on consumer hardware.

## CHAPTER 4 RESULTS

The results of the project should be understood across several dimensions: system implementation, offline model behaviour, deployment performance, and live usability. Taken together, these results show that the project successfully produced an end-to-end ASL recognition and translation prototype that operates in real time on standard hardware and supports a substantially larger vocabulary than many small-scale webcam-based demonstration systems.

The first major result is the successful implementation of the full system pipeline. The system includes a real-time application entry point, reusable modules for keypoint extraction, sequence modelling, text-to-speech, user interface, and video capture, as well as shared preprocessing utilities and training scripts. This implementation result matters because it demonstrates that the project was realized as a complete working system rather than remaining at the level of conceptual design. The pipeline integrates webcam acquisition, MediaPipe landmark extraction, frame normalization, temporal sequence buffering, model inference, prediction stabilization, on-screen visualization, and spoken output. In other words, the project achieved the architectural objective it set for itself.

The second major result concerns data and model infrastructure. The project successfully supported a WLASL300 workflow that includes label-map preparation, shared preprocessing, training, evaluation, and experiment management. This is significant because real-time recognition quality depends on strong alignment between offline data preparation and runtime feature generation. In the active WLASL300 path, `wlasl_runtime_matches_training = true` and `max_abs_diff_wlasl_runtime_vs_training = 0.0`. This exact match means that when the same landmark sequence is passed through the runtime path and the training path, the resulting engineered features are numerically identical. By contrast, `max_abs_diff_legacy_main_vs_training = 1.2382903099060059`, which shows that the older legacy path is not parity-equivalent and should not be used to claim training-runtime consistency. This result is technically important because it rules out hidden feature mismatch as a likely explanation for the remaining live errors.

The third major result lies in the model and experiment outputs. The project explored multiple model variants, including hand-only input, pose-enhanced input, and pose-plus-face input with stronger augmentation. The held-out evaluation summarized in the current classification report produced an overall sample-level accuracy of `0.5195` on `668` samples, with macro-average precision `0.5321`, macro-average recall `0.5276`, macro-average F1-score `0.4954`, weighted-average precision `0.5354`, and weighted-average F1-score `0.4931`. These numbers mean that the model was correct on a little over half of held-out Top-1 predictions overall, while class-balanced performance remained lower because the harder classes were not solved evenly. Although these figures do not represent state-of-the-art large-scale recognition, they are meaningful within the project’s constraints because they show that a lightweight mid-vocabulary sign classifier can learn substantial structure across 300 classes while remaining deployable.

These figures should be interpreted relative to the difficulty of the task and the project’s chosen design goals. A 300-class isolated sign problem with signer variation is substantially more challenging than alphabet or digit recognition, and a project optimized simultaneously for consumer-hardware inference, real-time responsiveness, and speech output will not necessarily reach the same benchmark ceiling as heavier research systems optimized only for accuracy. The results therefore demonstrate that the project succeeded in prioritizing deployability without collapsing into an oversimplified low-class-count demo. That balance is itself a meaningful outcome.

The experiment history gives a clearer picture of how the final figures were reached. The hand-only high-capacity run `asl_model_300_bilstm512_plateau_v1` used only hand landmarks and a larger recurrent state; it reached `60.67%` validation Top-1, `57.50%` test Top-1, and `83.57%` test Top-5. The pose-enhanced run `asl_model_300_bilstm512_pose_v1` added selected upper-body pose joints and improved these figures to `68.00%` validation Top-1, `63.03%` test Top-1, and `84.59%` test Top-5. The longer-window pose run `asl_model_300_bilstm512_pose_seq40_targeted_v1` increased the temporal window to `40` frames in order to preserve more motion detail for difficult signs; it reached `68.10%` validation Top-1, `61.82%` test Top-1, and `85.80%` test Top-5. The earlier face-aware balanced-augmentation run `asl_model_300_pose_face_balaug_v1` added compact face anchors and stronger balancing-plus-augmentation choices; it reached `67.46%` validation Top-1, `88.90%` validation Top-5, `60.03%` test Top-1, and `85.80%` test Top-5. The final hardened face-aware run `asl_model_300_pose_face_balaug_hardened_v1` retained the face-aware `30 x 180` representation but strengthened the training recipe and live deployment alignment, reaching `72.31%` validation Top-1, `89.33%` validation Top-5, `65.01%` test Top-1, and `87.59%` test Top-5.

Table 1 summarizes the main model-development stages.

| Run | Main change | Val Top-1 | Val Top-5 | Test Top-1 | Test Top-5 |
|---|---|---:|---:|---:|---:|
| `asl_model_300_bilstm512_plateau_v1` | Hand-only high-capacity baseline | 60.67% | not recorded here | 57.50% | 83.57% |
| `asl_model_300_bilstm512_pose_v1` | Added selected pose joints | 68.00% | not recorded here | 63.03% | 84.59% |
| `asl_model_300_bilstm512_pose_seq40_targeted_v1` | Longer 40-frame pose window | 68.10% | 88.47% | 61.82% | 85.80% |
| `asl_model_300_pose_face_balaug_v1` | Added compact face anchors and stronger augmentation | 67.46% | 88.90% | 60.03% | 85.80% |
| `asl_model_300_pose_face_balaug_hardened_v1` | Hardened face-aware final run | 72.31% | 89.33% | 65.01% | 87.59% |

These experiments also show what changed from one stage to the next. The hand-only model tested whether a strong recurrent backbone alone could solve the task with minimal inputs. The pose-enhanced run tested whether body-relative context would help separate signs whose hand shapes look similar but occur in different locations. The longer-window pose run tested whether extra temporal context would help motion-sensitive confusions. The face-aware balanced-augmentation run tested whether compact head-relative anchoring plus stronger augmentation would improve signs articulated near the face. The final hardened run kept the best parts of that face-aware design while tightening the training schedule and strengthening the deployment path that was actually used in live testing.

Detailed offline analysis provides further insight into class-specific behaviour. The weakest classes on the held-out split included `BEFORE`, `WHO`, `FINE`, `THIN`, `CORN`, `PINK`, `BIRTHDAY`, `BACKPACK`, `BAR`, `CHECK`, and `FAR`, all of which had recall `0.000` in the summarized weak-class list. Recurrent misclassification pathways included `CHAIR -> TRAIN`, `DEAF -> GOVERNMENT`, `KISS -> MORE`, `DOCTOR -> LEARN`, `BUT -> DIFFERENT`, `SON -> DAUGHTER`, `BAR -> YESTERDAY`, `CHECK -> LETTER`, `MOVIE -> BUSINESS`, and `SALT -> TRAIN`. These results show that the model’s errors are not random. Rather, they cluster around visually similar or temporally overlapping sign pairs, as would be expected in a large isolated-sign vocabulary. This structured error behaviour is important because it makes targeted improvement feasible.

Representative strong classes and the complete weakest-class subset from the held-out analysis are summarized in Table 2.

| Class group | Gloss | Precision | Recall | F1 | Support | Main note |
|---|---|---:|---:|---:|---:|---|
| Strong | HELP | 1.000 | 1.000 | 1.000 | 3 | Fully recovered on held-out split |
| Strong | NO | 1.000 | 1.000 | 1.000 | 3 | Fully recovered on held-out split |
| Strong | TABLE | 1.000 | 1.000 | 1.000 | 3 | Fully recovered on held-out split |
| Strong | HOW | 0.750 | 1.000 | 0.857 | 3 | High recall with small support |
| Strong | MANY | 0.750 | 1.000 | 0.857 | 3 | High recall with small support |
| Weak | BEFORE | 0.000 | 0.000 | 0.000 | 4 | Often confused with `FATHER` |
| Weak | WHO | 0.000 | 0.000 | 0.000 | 3 | Most often predicted as `RUSSIA` |
| Weak | FINE | 0.000 | 0.000 | 0.000 | 3 | Most often predicted as `MAN` |
| Weak | THIN | 0.000 | 0.000 | 0.000 | 3 | Most often predicted as `SOON` |
| Weak | CORN | 0.000 | 0.000 | 0.000 | 3 | Most often predicted as `BALL` |
| Weak | PINK | 0.000 | 0.000 | 0.000 | 3 | Most often predicted as `HARD` |
| Weak | BIRTHDAY | 0.000 | 0.000 | 0.000 | 3 | Most often predicted as `ACCIDENT` |
| Weak | BACKPACK | 0.000 | 0.000 | 0.000 | 3 | Most often predicted as `DOOR` |
| Weak | BAR | 0.000 | 0.000 | 0.000 | 3 | Often confused with `YESTERDAY` |
| Weak | CHECK | 0.000 | 0.000 | 0.000 | 3 | Often confused with `LETTER` |
| Weak | FAR | 0.000 | 0.000 | 0.000 | 3 | Most often predicted as `BEHIND` |
| Weak | KISS | 0.250 | 0.333 | 0.286 | 3 | Repeated overlap with `MORE` |
| Weak | DRINK | 0.500 | 0.250 | 0.333 | 4 | Low recall despite higher support |
| Weak | YES | 0.200 | 0.333 | 0.250 | 3 | Most often predicted as `EAST` |
| Weak | LETTER | 0.250 | 0.333 | 0.286 | 3 | Most often predicted as `HEADACHE` |

Table 3 summarizes the most frequent confusion pairs in the held-out analysis.

| True gloss | Predicted gloss | Count | Category | Suggested direction |
|---|---|---:|---|---|
| CHAIR | TRAIN | 2 | Temporal Confusion | More motion-aware training, longer windows, pose cues |
| DEAF | GOVERNMENT | 2 | Temporal Confusion | More motion-aware training, longer windows, pose cues |
| KISS | MORE | 2 | Temporal Confusion | More motion-aware training, longer windows, pose cues |
| DOCTOR | LEARN | 2 | Temporal Confusion | More motion-aware training, longer windows, pose cues |
| BUT | DIFFERENT | 2 | Temporal Confusion | More motion-aware training, longer windows, pose cues |
| SON | DAUGHTER | 2 | Temporal Confusion | More motion-aware training, longer windows, pose cues |
| BAR | YESTERDAY | 2 | Temporal Confusion | More motion-aware training, longer windows, pose cues |
| CHECK | LETTER | 2 | Temporal Confusion | More motion-aware training, longer windows, pose cues |
| MOVIE | BUSINESS | 2 | Temporal Confusion | More motion-aware training, longer windows, pose cues |
| SALT | TRAIN | 2 | Temporal Confusion | More motion-aware training, longer windows, pose cues |

Rather than relying on a large confusion-matrix image that is difficult to read and only summarizes a subset of classes, this report focuses on the more interpretable weak-class and confusion-pair summaries in Tables 2 and 3.

The error structure was also specific rather than diffuse. `BEFORE` had support `4` with recall `0.000`, `BAR` had support `3` with dominant confusion into `YESTERDAY`, and `CHECK` had support `3` with dominant confusion into `LETTER`. High-frequency pairs such as `CHAIR -> TRAIN`, `KISS -> MORE`, `DOCTOR -> LEARN`, `BUT -> DIFFERENT`, and `CHECK -> LETTER` were categorized as temporal confusions, reinforcing the conclusion that motion modelling and body-relative context remained central weaknesses.

Another key result is the success of the real-time runtime design. The system is able to run webcam-based inference, maintain a rolling temporal sequence, generate top-k predictions, and stabilize those predictions sufficiently for live use. Confidence squelching means that low-confidence guesses are discarded before they can affect the stabilized output. Adaptive thresholds mean that a few signs can still be accepted below the global threshold if they remain clearly ahead of their rivals. Motion requirements mean that selected signs are blocked when the hands are too static, preventing false firings for motion-dependent classes such as `DOCTOR`, `HEARING`, or `TEST`. Per-sign overrides mean that specific signs such as `APPROVE`, `ARRIVE`, `BOOK`, `FINE`, `HOPE`, `JACKET`, and `LAW` can use slightly lower thresholds when live evidence shows they are correct but systematically under-confident. Confusion suppression means that output is withheld when a known rival, such as `MORE` during `KISS` or `DIFFERENT` during `BUT`, remains too close in score. Together, these controls show that the project did not treat recognition as a raw frame-level classification problem, but as an interactive temporal decision problem. This produced a recognizer that, while still imperfect, could be used in direct live testing and demonstration rather than only evaluated in batch mode.

The runtime design also demonstrates that practical machine learning applications depend on post-model logic. In the context of this project, the neural network is the central predictor, but not the sole determinant of user-visible behaviour. The final output depends on how predictions are buffered, filtered, stabilized, and translated into text and speech. This is an important result because it shows that substantial gains in practical usability can be achieved without replacing the model entirely. Instead, improvements in calibration and runtime decision policy can convert unstable predictions into more trustworthy interaction.

Buffering in this system means that the runtime continuously keeps the latest `30` frames of preprocessed features rather than classifying isolated frames. Filtering means that predictions can be suppressed if they fail confidence, margin, motion, or confusion checks. Stabilization means that accepted candidates are aggregated across a `10`-prediction history and must normally collect at least `6` votes before they become the displayed sign; in other words, the same sign must appear consistently over most of the recent prediction history before it is trusted. Translation into text and speech then happens only after this stabilized result is available.

Several post-model improvements were introduced over time. Idle-state clearing was added so stale predictions disappear after signing stops instead of lingering on screen. The `10`-frame grace period was added so brief hand dropouts no longer clear the temporal buffer immediately. Adaptive lower-confidence acceptance was added for signs that were often correct in live use but consistently below the global threshold. Confusion-pair suppression was added so near-tied rival signs do not trigger premature output. Peak-sign preservation was added so short strong peaks such as `ARRIVE`, `CATCH`, `HOPE`, `JACKET`, and `LAW` remain visible even if the end pose becomes noisier. TTS hysteresis was also added so repeated speech fires only when the stabilized sign changes, preventing distracting repeated utterances while the same sign is being held.

The live-testing results are one of the most practically significant outcomes of the project, but in the context of this report they serve as evidence about the deployed system rather than as the sole subject of investigation. The live test covered the 300-sign vocabulary and recorded whether each gloss was successfully recognized, recognized after refinement, present only in the top five, absent from the top five, or still pending due to runtime issues. Initial live-testing records showed 35 signs appearing in the top five but not being finalized, 9 signs not appearing in the top five at all, and 8 pending cases. This corresponded to an operational success rate of 82.7% across the 300 signs. After iterative refinement, the unresolved cases fell to 22 top-five failures, 1 complete top-five absence, and 2 pending signs, raising live operational success to 91.67%.

This live improvement is best interpreted as evidence that the full system architecture is workable and that many deployment issues are tractable. A sign that is initially missed but later recognized after refined timing, visibility, or motion handling indicates that the underlying representation already contains useful class information. It also indicates that runtime calibration can have a major impact on practical usability. The shift from 82.7% to 91.67% therefore demonstrates more than user adaptation alone; it demonstrates that the system design supports meaningful improvement through debugging and refinement rather than requiring complete retraining for every problem.

The refinement process itself involved targeted engineering changes rather than vague retesting. The first part of the refinement was model-side selection. The live path was moved onto the stronger hardened face-aware checkpoint because earlier variants left too many signs either under-confident or overly dependent on hand shape alone. This shift gave the live system access to the improved `30 x 180` representation and the better validation and test figures of the hardened run rather than the weaker earlier checkpoints.

The second part of the refinement was threshold and stabilization tuning. Live testing showed that some signs were already reaching the correct label, but not strongly enough or not for long enough to survive the stabilization rules. To address this, adaptive thresholds were introduced for repeatedly under-confident but often-correct signs such as `APPROVE`, `BOOK`, `FINE`, `HOPE`, `JACKET`, and `LAW`. Confusion-pair suppressors were then layered on top so that signs already known to compete closely, such as `KISS/MORE` or `BUT/DIFFERENT`, would not finalize too early when the margin remained weak. Peak-sign preservation was added for signs such as `ARRIVE` and `CATCH`, where the correct class could spike briefly at the expressive part of the motion and then be overwritten by a misleading end pose.

The third part of the refinement focused on live robustness rather than raw class scores. The `10`-frame grace period was added so brief hand-loss events no longer wiped out otherwise valid sequence context. Idle-state clearing and low-motion handling were added after false activations such as `LATE` appeared when the user was not signing. Idle-state clearing means that once the hands disappear for long enough, or once the scene no longer contains meaningful signing motion, the runtime actively clears stale vote history, current candidate state, and the carried context instead of letting an old sign linger on screen and in speech. Low-motion handling means that if the hands remain visible but move too little, the system treats that state as insufficient evidence rather than as a completed sign. This was important because some false outputs were caused not by a completely wrong dynamic sequence, but by near-static postures or partial gestures that accidentally resembled a known class. Motion-aware blocking was therefore tightened for signs like `DOCTOR`, `HEARING`, `CEREAL`, `EAT`, and `TEST`, where static posing or partial motion could still trigger the class incorrectly. Together, these changes made the live loop less eager to speak a sign unless the movement pattern matched the intended gloss more faithfully.

The final part of the refinement concerned user execution. Live notes showed that some failures were not complete recognition breakdowns but mismatches between how the user performed the sign and how the model had learned to expect it. As a result, articulation was adjusted sign by sign: some signs such as `DAY`, `DISAPPEAR`, and `REMEMBER` were slowed down so the model could accumulate a clearer temporal pattern; some signs such as `DANCE` and `RIDE` needed both hands more clearly visible at the same time; some signs such as `CHAIR`, `DELICIOUS`, and `TRAIN`, as well as signs like `ALL` that benefited from slightly greater camera distance, needed a clearer body-relative anchor; and others such as `BATHROOM`, `CAN`, and `GOVERNMENT` worked better when the wrist carried the motion instead of the whole arm. The improvement from `82.7%` to `91.67%` therefore came from a combination of stronger model choice, stricter runtime policy, and more informed live execution rather than from any single isolated tweak.

The live-testing log also generated rich qualitative results about how the system behaves. Many signs were recognized successfully once articulation became more aligned with the model’s learned assumptions. Some signs needed slower execution to accumulate enough temporal evidence; for example, `DAY`, `DISAPPEAR`, and `REMEMBER` were noted as more reliable when signed more slowly. Others needed clearer hand visibility, especially for both hands simultaneously; `DANCE` and `RIDE` are examples where clearer visibility of both hands improved success. Some required body-relative clarity such as being farther from the camera or signing from the side; `CHAIR`, `DELICIOUS`, and `TRAIN` benefited from clearer side presentation, while `ALL` was noted as easier when performed farther from the camera. Some required that movement be isolated to the wrist rather than the whole arm; `BATHROOM`, `CAN`, and `GOVERNMENT` were all noted as behaving better when the wrist carried the motion while the rest of the arm remained more stable. These outcomes suggest that the model is sensitive to specific motion and spatial cues, which is consistent with a keypoint-based temporal recognizer trained on a finite dataset.

Several signs were observed to behave especially well, with stable recognition and few special requirements. Many core vocabulary items across nouns, verbs, adjectives, and functional glosses were recognized successfully. This breadth of recognized vocabulary is itself an important result because it shows that the project did not merely succeed on a trivial subset of easy gestures. The deployed recognizer can handle a wide range of signs spanning different motion types and hand configurations, which validates the overall feasibility of a 300-class real-time system.

Offline artefacts also show clusters of comparatively strong classes. In the held-out classification report, signs such as `HELP`, `NO`, `TABLE`, `BED`, `CAN`, `CLASS`, `DAY`, and `YOUR` reached perfect or near-perfect precision-recall combinations within their support counts. This does not imply that all of them were equally easy in every live setting, but it does show that the model learned robust structure for a meaningful subset of the vocabulary rather than succeeding only on a handful of trivial gestures.

At the same time, the results reveal persistent categories of difficulty. One category consists of signs that appear in the top five predictions but do not become final output. This indicates partial recognition combined with insufficient confidence, instability, or confusion suppression; `FINE`, `SIGN`, `APPROVE`, and `HOPE` are representative examples, because they were often present or even briefly dominant but still failed to cross the final speaking threshold consistently. Another category consists of signs that are momentarily recognized but then replaced by another class once the gesture resolves into a static posture; `ARRIVE` dropping into `BABY`, `CATCH` dropping into `YEAR`, `CRASH` drifting into `PROBLEM`, and `LAW` being overtaken after the gloss ends are clear examples of end-pose takeover. A third category involves signs that trigger incorrectly under incomplete or static conditions, showing that motion gating remains imperfect for some classes; this was observed for `DOCTOR`, `HEARING`, `CEREAL`, `EAT`, and `TEST`, where the system could respond to incomplete motion or even a held pose. These findings matter because they point to deployment-specific refinement needs rather than simple model collapse.

The live results align in meaningful ways with the offline artefacts. Signs such as BEFORE, FINE, CORN, FAR, and KISS appear both in weak-class summaries and in live-testing difficulties. Confusions such as KISS with MORE and BUT with DIFFERENT appear in both sources as well. This alignment is an important project-level result because it shows coherence between controlled evaluation and real-world behaviour. The live system’s errors are often direct expressions of the same underlying class ambiguity visible in offline testing.

Another important result is that the project achieved usable text-to-speech integration. Recognized signs can be emitted as spoken words, making the prototype more meaningful as an accessibility tool. The availability of both on-screen and audible output supports a more direct sign-to-hearing communication workflow than recognition alone would offer. While the system remains limited to isolated glosses rather than full sentence generation, the TTS module demonstrates that the project successfully crossed the boundary from classification toward communication support.

This transition from recognition to communication is worth emphasizing. Several systems discussed in the literature emphasize recognition accuracy or classification output without extending the interaction into a spoken communication layer (Alsharif et al., 2025; Kamble, 2025; Uddin et al., 2025). By incorporating speech output, the project reframed recognition as a communicative action. Even when limited to isolated words, that difference changes the value of the prototype because the output becomes immediately usable by a hearing conversation partner rather than remaining a label visible only on the screen. It becomes easier to imagine how the system might support real users in controlled contexts such as educational demonstrations, basic requests, or proof-of-concept accessibility interactions.

The project also produced several technical reporting artefacts that are themselves meaningful outputs. These include classification reports, confusion pairs, actionable fix summaries, parity checks, weak-class summaries, and confusion matrices. Such artefacts are valuable because they formalize the model’s strengths and weaknesses and create a foundation for future refinement. In a final-year project context, producing a system together with systematic diagnostic evidence is a stronger outcome than producing a system alone.

These artefacts served different roles. `classification_report.txt` and `classification_report.json` quantified per-class precision, recall, F1-score, and support for the full held-out split, with the complete values reproduced in Appendix A. `weak_classes_summary.txt` isolated the lowest-recall classes with enough support to matter and also recorded their most common mistaken targets, which is why Table 2 can report, for example, that `BAR` was most often absorbed by `YESTERDAY` and `CHECK` by `LETTER`. `confusion_pairs.txt` listed the most frequent directional error pathways, which are reproduced in Table 3. `actionable_fixes.json` translated those pathways into concrete next-step suggestions such as stronger motion-aware training or richer pose cues for temporal confusions. `preprocessing_parity.json` checked agreement between the runtime and training feature pipelines by comparing the engineered feature outputs produced by each path for the same input sequence. On the active WLASL300 path the comparison returned `wlasl_runtime_matches_training = true` and `max_abs_diff_wlasl_runtime_vs_training = 0.0`, showing exact equality, while the older legacy path differed by `max_abs_diff_legacy_main_vs_training = 1.2382903099060059`. Together, these artefacts formed the main evidence base for the results presented in this report.

The results can also be evaluated against the original project objectives. The first objective, to build a real-time pipeline using standard hardware and a regular webcam, was achieved. The project uses webcam input and MediaPipe-based extraction without dependency on specialized sensors. The second objective, to train a model for 300-sign recognition with suitable preprocessing, was also achieved. The system includes normalization, feature engineering, class balancing options, and a trained temporal classifier, and the held-out and live evaluations both showed meaningful performance. The third objective, to deliver low-latency translated output through text and speech, was achieved in prototype form through the integrated real-time interface and TTS component. While some signs still require calibration and the project does not yet achieve unrestricted robustness, the core objectives were met.

From a deliverables standpoint, the project produced not only a working prototype but also reusable technical assets. These include training scripts, evaluation tools, runtime modules, reports, confusion analyses, and configuration pathways for future experiments. This is an important project result because it means the work can be extended rather than restarted. Future improvements can build directly on the existing infrastructure, which is a sign of sound software engineering and a strength in the context of academic project work.

The most important overall result, then, is not any single percentage. It is the demonstration that a mid-vocabulary, webcam-based, real-time ASL recognition and sign-to-speech system can be built, trained, deployed, and used meaningfully on consumer hardware with a largely keypoint-based approach. The project has therefore validated its central design hypothesis.

## CHAPTER 5 DISCUSSION

The results show that the project occupies an important middle ground within the sign-language technology landscape. On one side are research-heavy systems that pursue maximum recognition or translation performance through raw video modelling, large multimodal corpora, and computationally expensive architectures (Camgoz et al., 2020; Duarte et al., 2021; Woods & Rana, 2023a). On the other side are lightweight demonstration systems that run in real time but recognize only alphabets, digits, or very small gesture sets (Alsharif et al., 2025; Kamble, 2025; Uddin et al., 2025). This project positions itself between those extremes. It does not attempt continuous sentence translation, but it significantly expands beyond toy-vocabulary prototypes. It does not rely on expensive hardware, yet it supports a 300-sign vocabulary and spoken output. This middle-ground positioning is one of the project’s most important contributions.

From a problem-solving perspective, the project addresses the accessibility barrier identified in the problem statement in a practical and defensible way. It does not claim to eliminate all communication barriers between deaf and hearing individuals. Rather, it demonstrates a feasible method of reducing that barrier in contexts where isolated signs can be recognized and converted into spoken output on commonly available equipment. This matters because assistive technology often becomes useful not by solving the whole problem perfectly, but by making a meaningful portion of the problem more manageable in real settings.

The literature reviewed for the project helps explain why the chosen methodology was appropriate. Li et al. (2020) showed that large-vocabulary sign recognition is difficult even under controlled benchmark conditions, which supports the decision to treat WLASL300 as a challenging but appropriate target. Kamble (2025), Uddin et al. (2025), and Alsharif et al. (2025) reinforced the feasibility of MediaPipe-plus-LSTM style pipelines for real-time tasks on standard hardware, which justifies the project’s core engineering direction. Camgoz et al. (2020) and Duarte et al. (2021) demonstrate the value of richer temporal and multimodal modelling, but they also highlight the complexity and resource demands of more ambitious continuous translation systems. Tan et al. (2024) and Holmes et al. (2024) support the strategic value of keypoint-based efficiency and focused feature design. Seen in this context, the project’s architecture is not arbitrary; it is a direct response to the strengths and limitations documented in prior work.

The overall design is strongest when viewed as a full recognition pipeline rather than as a model in isolation. WLASL300 provided a benchmark with enough signer and vocabulary variability to make the task meaningful. MediaPipe made real-time keypoint extraction feasible on ordinary hardware. Temporal sequence modelling addressed the fact that sign identity depends on motion rather than a single frame. Normalization, landmark selection, and confusion analysis then helped convert those ingredients into a system that could be interpreted, debugged, and refined in deployment.

One of the strongest aspects of the project is this systems-level integration. The deployed recognizer does not rely on the neural network alone; it combines preprocessing parity, top-k prediction handling, runtime stabilization, confusion suppression, and text-to-speech output. In real-time sign recognition, output timing, false activations, label persistence, and user feedback are central to usability. The live-testing outcomes show that these layers mattered directly. A sign that appeared correctly in the top five but never collected enough votes was not yet useful, and a sign that peaked correctly but was replaced by a static end pose required peak preservation to remain usable. Likewise, idle false positives such as the early `LATE` prediction showed that live trustworthiness depends on runtime controls as much as on classifier quality.

The decision to use a BiLSTM with attention is also defensible when considered against project constraints. Transformer-based architectures have shown strong sequence modelling performance in sign-language research, especially in translation-oriented settings (Camgoz et al., 2020; Woods & Rana, 2023a). However, they are often heavier and more demanding in terms of data and inference cost. For a mid-scale isolated sign task intended for consumer hardware deployment, a BiLSTM provides a strong balance between temporal modelling power and implementation practicality (Kamble, 2025; Uddin et al., 2025). The project results suggest that this balance was reasonable. Although the model does not fully solve all difficult class confusions, it captures enough temporal structure to support meaningful live recognition across a large vocabulary.

The use of landmarks rather than full RGB frames is perhaps the most consequential design decision in the project. This choice reduces dimensionality, improves efficiency, and makes real-time deployment much more achievable (Holmes et al., 2024; Tan et al., 2024). It also aligns with broader trends identified in the literature toward pose-based deployable systems (De Coster et al., 2023; Holmes et al., 2024; Tan et al., 2024). The project results support this choice strongly. The system is able to operate in real time, and its errors are generally interpretable in terms of motion, orientation, and body-relative structure rather than opaque visual texture issues. This interpretability is valuable. When a sign fails because the movement is too fast, too static, too close to the camera, or too similar to a neighbouring gloss, the problem can be investigated and often corrected. For example, `KISS` and `MORE` repeatedly overlapped in both offline and live results, `MOTHER` versus `FATHER` benefited from additional pose and face context, and short signs such as `ARRIVE` or `CATCH` needed peak preservation so a correct brief motion was not overwritten by its end pose. Such interpretability is more difficult in heavy RGB-based models (Camgoz et al., 2020; Duarte et al., 2021; Tan et al., 2024).

At the same time, the project results also reveal the limitations of a largely landmark-based approach. Several persistent confusions suggest that some signs differ in ways that may not be fully captured by the current feature representation or current temporal decision policy. Contact events, subtle palm orientation differences, and transitions into or out of static poses may be underrepresented in purely coordinate-driven modelling. This can be seen in pairs such as `KISS` versus `MORE`, `BUT` versus `DIFFERENT`, and `DOCTOR` versus `LEARN`, where the overall motion pattern is similar but finer relational cues still matter. It can also be seen in signs whose end pose resembles another class strongly enough to take over the prediction once movement stops. This is where the extended use of pose and compact face information becomes significant. The project already acknowledges that hands alone may not be sufficient for all classes. Future improvements may require deeper use of multimodal cues or more refined landmark relationships rather than simply more training epochs.

The discrepancy between offline held-out accuracy and live operational success deserves careful interpretation. The held-out classification results show more modest sample-level Top-1 performance than the final live-testing success rate. This difference should not be read as inconsistency or error. Rather, it reflects different evaluation questions. Offline metrics ask whether the model predicts the correct label for unseen dataset samples under a strict benchmark protocol. Live operational success asks whether the deployed system can be made to recognize the intended sign during interactive use, aided by prediction stabilization, repeated attempts, and execution adjustments. The live metric therefore measures practical usability under informed operation, not strict generalization in the same sense as the held-out dataset score. Both metrics matter, but they must not be conflated.

This distinction leads to a broader methodological insight. Real-time sign recognition evaluation should ideally separate unaided performance from informed operational performance. The present project demonstrates strong informed operational performance: once the system’s behaviour is understood and refined, many signs can be recognized successfully. This is valuable because it shows that the architecture is viable. However, future work should also quantify first-attempt naive usage, because that would better estimate how the system performs for new users who have not learned its preferred articulation patterns. The current project lays the groundwork for such future evaluation by already identifying which signs are most sensitive to pace, viewpoint, visibility, and motion gating, including short peak-dependent signs such as `ARRIVE` and `JACKET`, face-anchored distinctions such as `MOTHER` and `FATHER`, and motion-sensitive confusions such as `BUT` and `DIFFERENT`.

The live-testing findings further reveal that the remaining problems are concentrated, not diffuse. This is encouraging. If errors were random across the vocabulary, improving the system would require a more fundamental redesign. Instead, the project shows recurring patterns. Top-five-but-not-finalized cases are signs that appear among the strongest candidates but never collect enough stabilized votes to become the spoken output; examples include `FINE`, `SIGN`, and `APPROVE`. End-pose takeover refers to situations in which the motion of a sign is predicted correctly for a brief moment, but the final held posture resembles another class and replaces it; examples include `ARRIVE` being replaced by `BABY`, `CATCH` being replaced by `YEAR`, and `LAW` losing its correct spike after the sign finishes. Incomplete-motion triggering refers to false activations caused by a partial or idle pose before the sign has actually been completed; examples include `DOCTOR`, `HEARING`, and `TEST` activating too easily from insufficient motion. Confusion among a relatively small set of similar signs includes recurring pairs such as `KISS` and `MORE`, `BUT` and `DIFFERENT`, or `DOCTOR` and `LEARN`. This means future work can be strategic. Specific signs or sign pairs can be targeted through class-specific thresholds, better onset-offset detection, stronger motion requirements, richer feature cues, or additional training examples.

The project also contributes a useful perspective on evaluation interpretation. In sign-language technology, there is a temptation either to foreground only benchmark scores or to foreground only persuasive demo behaviour. This project illustrates why both are needed and why they must be read carefully. Offline evaluation exposes systematic weaknesses such as low-recall classes and repeated confusion pathways. Live evaluation shows whether those weaknesses actually disrupt practical interaction and whether runtime interventions can compensate for them. Neither view alone is complete. A system with better held-out accuracy but poor live stability may still feel unusable, while a system with good demo behaviour but weak benchmark generalization may not scale beyond familiar users or curated conditions. The project’s combined evidence therefore supports a more balanced evaluation philosophy for future sign-language research.

The project’s use of live-testing observations as engineering feedback is another strength. Some signs required slower execution so that enough discriminative frames entered the `30`-frame signing window; this was especially relevant for short peak-sensitive classes such as `ARRIVE` and `CATCH`. Others benefited from clearer elbow visibility or a slightly different camera distance, particularly when body-relative position helped distinguish nearby signs such as `MOTHER` and `FATHER`. Others needed stronger wrist isolation so that hand motion was captured cleanly instead of being blurred by larger arm movement; `BATHROOM`, `CAN`, and `GOVERNMENT` are examples where live notes specifically emphasized wrist-led motion. These findings show that the deployed recognizer has operational expectations that can be surfaced and analyzed. From a human-computer interaction standpoint, this implies two possible development directions. One direction is to make the model more robust so that it tolerates natural variation better. The other is to make the interface more instructional, teaching users how to position themselves and perform signs more effectively. In the short term, both directions have value. An assistive system can become more useful immediately through clear user guidance even while underlying robustness continues to improve.

Another important discussion point concerns the scope of isolated sign recognition itself. The project intentionally limits itself to isolated signs rather than full continuous translation. This limitation should not be treated merely as a shortcoming. It is also a methodological choice that enabled the project to achieve a meaningful result within available resources. Continuous sign language translation involves not only recognition of glosses, meaning dictionary-style label units used to represent signed words or concepts, but also segmentation, sequence alignment, contextual language modelling, and often far heavier computation (Camgoz et al., 2020; Duarte et al., 2021; Tan et al., 2024). Segmentation means detecting where one sign begins and ends inside a continuous stream. Sequence alignment means matching changing video frames to the intended series of glosses across time. Contextual language modelling means using neighbouring signs and sentence structure to infer which interpretation makes sense in context. By focusing on isolated signs, the project concentrated on building a robust foundation: reliable keypoint extraction, stable temporal classification, and usable real-time output. In that sense, the project establishes a platform from which more advanced continuous or context-aware work could grow later.

The societal significance of the project also deserves discussion. A webcam-based ASL sign-to-speech prototype that runs on ordinary hardware is inherently more deployable than systems depending on specialized sensors or laboratory setups (Alsharif et al., 2025; Gan et al., 2023; Tan et al., 2024). This matters for inclusion because accessibility technologies only create broad impact when they can be used in everyday environments. Potential use cases include demonstrations in educational settings, accessibility prototypes for service counters, or assistive interfaces for basic word-level communication. While the current project is not yet a finished commercial-grade assistive product, it moves in a direction that is more socially practical than many research-only systems.

The project also has pedagogical significance. Because the system produces visible and audible outputs from human movement, it can help communicate the challenges and possibilities of sign-language AI to non-specialist audiences. This makes it useful not only as a prototype application, but also as a teaching tool in discussions of machine learning, accessibility, human-computer interaction, and responsible technology design. The combination of a concrete social problem and a technically tractable implementation makes the project especially valuable in an educational context.

This educational value should not be underestimated. Projects of this kind help bridge the gap between abstract machine learning theory and socially grounded system design. They require decisions about model architecture, preprocessing, evaluation, interface behaviour, and ethical framing all at once. As a result, the project demonstrates not only technical competence in deep learning and software integration, but also an ability to reason about accessibility, user needs, and deployment realism. That broader integration of concerns is one of the clearest indicators that the project has value beyond a narrow coding exercise.

From an academic perspective, the project contributes by integrating multiple strands of current sign-language research into a coherent applied system. It takes the dataset realism of WLASL, the efficiency of MediaPipe keypoints, the temporal power of LSTM-based modelling, the deployment lessons of real-time webcam systems, and the diagnostic focus of confusion analysis, then combines them into a single project. This integration itself is valuable. It shows that student-level implementation can still engage seriously with the state of the field while producing a working prototype rather than merely summarizing prior work.

Several limitations remain evident. The system is currently restricted to isolated signs rather than continuous sentences. Live performance still depends partly on informed articulation and runtime calibration. Some classes remain weak both offline and online. The project does not yet appear to include a large multi-user usability study, so generalization across diverse signers remains only partially assessed. Latency, while designed for real-time use, could be profiled more explicitly across different hardware conditions. And because the system maps single recognized glosses to speech, it does not yet address grammar, context, or sentence-level translation.

There are also limitations inherent to the dataset and problem framing. WLASL, while valuable and widely used, represents isolated glosses rather than spontaneous conversational signing (Li et al., 2020). In this context, a gloss is the label assigned to an individual sign concept, such as a word-level target in the dataset. A model trained in this setting may learn to recognize many lexical items while still struggling with coarticulation, discourse context, and continuous movement boundaries found in real communication (Camgoz et al., 2020; Duarte et al., 2021). The present project is therefore best understood as a strong isolated-sign platform rather than a complete sign-language translation solution. Recognizing this distinction is important for interpreting both the successes and the limitations of the results.

Even with these limitations, the project outcome remains strong. The combination of mid-scale vocabulary, real-time webcam operation, keypoint efficiency, and speech output is precisely what much of the literature identifies as difficult to achieve simultaneously (Gan et al., 2023; Holmes et al., 2024; Tan et al., 2024). The project does not solve the entire sign-language processing problem, but it validates a promising and practical region of the design space. In that sense, it succeeds both as a research-informed engineering project and as a meaningful accessibility-oriented prototype.

## CHAPTER 6 CONCLUSION AND RECOMMENDATION

This project set out to address a clear and important problem: the lack of a lightweight, real-time, webcam-based ASL system that can recognize a substantial vocabulary and translate recognized signs into spoken output on standard consumer hardware. The resulting system demonstrates that such a pipeline is feasible. By combining MediaPipe landmark extraction, normalization and feature engineering, a BiLSTM-based temporal classifier, prediction stabilization, and text-to-speech output, the project successfully produced an end-to-end prototype for isolated ASL sign recognition and translation.

The report has shown that the project should be understood as a whole-system achievement rather than only as a live-testing exercise. Its contributions include the selection and preparation of WLASL300 as a suitable mid-scale dataset, the implementation of a training and evaluation workflow, the creation of a real-time runtime pipeline with stabilization logic, the integration of spoken output, and the generation of offline and live performance evidence. Together, these outcomes validate the project’s core hypothesis that a keypoint-based, temporally modelled sign recognizer can provide meaningful real-time functionality without specialized hardware.

The offline results demonstrate that the model learned substantial structure across 300 classes, even though large-vocabulary sign recognition remains difficult. The presence of structured confusion patterns and weak-class summaries shows that the system’s limitations are diagnosable rather than random. The live results are especially encouraging: operational success improved from 82.7% to 91.67% after iterative refinement, confirming that the deployed architecture is practically workable and responsive to targeted improvement. The project therefore succeeded in meeting its major objectives of real-time webcam-based recognition, 300-sign modelling, and text-and-speech output.

Several conclusions follow from these findings. First, pose- and landmark-based pipelines are a strong choice for deployable sign-language systems under consumer hardware constraints (Holmes et al., 2024; Tan et al., 2024). Second, temporal modelling remains essential even for isolated signs because meaningful recognition depends on motion rather than static hand configuration alone (Camgoz et al., 2020; Kamble, 2025; Uddin et al., 2025). Third, preprocessing consistency between training and runtime is a critical engineering requirement and was handled successfully in this project (De Coster et al., 2023). Fourth, real-time usability depends heavily on runtime decision logic such as confidence thresholds, motion gating, and confusion suppression (Gan et al., 2023; Kamble, 2025). Fifth, live testing is indispensable because many of the most important usability issues do not appear clearly in offline accuracy numbers alone.

The project also highlights a practical lesson for sign-language technology more broadly. There is substantial value in targeting the underexplored space between toy demonstrations and heavy research systems (Gan et al., 2023; Holmes et al., 2024; Tan et al., 2024). A 300-sign isolated ASL recognizer with speech output is not the final answer to sign-language translation, but it is large enough to be meaningful and lightweight enough to be deployable. That combination gives the project both practical relevance and future extensibility.

Another broader lesson is that accessibility-oriented AI projects benefit from integrated evaluation. A purely technical metric can miss human-facing problems such as flicker, delayed output, confusing false positives, or lack of understandable feedback. Conversely, purely anecdotal demonstrations can overlook structural model weaknesses and fail to provide rigorous evidence. The present project benefited from combining both perspectives. Offline reports made weaknesses measurable, while live testing made them tangible. This hybrid evaluation model is worth preserving in future work (Gan et al., 2023; Holmes et al., 2024; Tan et al., 2024).

Several recommendations emerge directly from the project outcomes.

First, future work should continue improving the difficult-sign subset identified by both offline and live evaluation. Signs that repeatedly appear in the top five without finalization, or that trigger confusion with a small set of competitor classes, should be prioritized for targeted retraining, threshold calibration, and motion-policy refinement (Holmes et al., 2024; Woods & Rana, 2023b).

Second, the runtime pipeline should be extended with stronger gesture onset and offset detection. In this context, onset detection means identifying when a sign has genuinely begun, while offset detection means identifying when the meaningful movement has ended and the system should commit the result. Several live issues arose because the end pose or partial pose of a sign either suppressed a correct recognition or triggered an incorrect one. For example, brief correct peaks for signs such as `ARRIVE` or `CATCH` could be overwritten by a later static posture, while incomplete motion could also trigger an unintended class too early. More explicit segmentation or event-based firing logic would likely improve both stability and trustworthiness.

Third, further experimentation with pose and face cues should be conducted for signs whose meaning depends strongly on body-relative location or subtle contextual geometry. The extended WLASL300 path already supports these features, and more systematic ablation studies could clarify where they help most (Duarte et al., 2021; Holmes et al., 2024).

Fourth, multi-signer evaluation should be expanded. The current project demonstrates feasibility convincingly, but broader signer diversity would better assess robustness to natural differences in articulation, pace, handedness, and body geometry.

Fifth, the project would benefit from more explicit latency benchmarking and runtime logging. Since low-latency usability is central to the project’s purpose, future versions should record detailed timing for capture, preprocessing, inference, stabilization, and text-to-speech triggering.

Sixth, a structured user guidance layer could improve practical accessibility even before model robustness is fully optimized. Guidance on camera framing, hand visibility, movement completion, and pacing would make the system easier to use in demonstrations and early real-world trials.

Seventh, the project provides a strong foundation for future progression toward phrase-level or continuous sign translation. While such expansion would require additional datasets, sequence segmentation, and language modelling, the present architecture already establishes key building blocks: keypoint extraction, temporal recognition, stable output logic, and speech integration (Camgoz et al., 2020; Duarte et al., 2021; Tan et al., 2024).

In conclusion, the project achieved a meaningful and technically sound result. It demonstrates that a real-time ASL recognition and sign-to-speech prototype with a 300-sign vocabulary can be built on accessible hardware using a keypoint-based temporal learning approach. Although challenges remain, especially around difficult classes, motion gating, and broader generalization, the system already shows strong potential as both a final-year engineering achievement and a foundation for future assistive sign-language technology. The project therefore makes a credible contribution to the ongoing effort to build more inclusive, practical, and deployable communication tools for deaf and hard-of-hearing users (Gan et al., 2023; Holmes et al., 2024; Tan et al., 2024).

More broadly, the project shows that progress in assistive AI does not always require the heaviest models or the richest sensing platforms. Careful problem scoping, appropriate dataset choice, efficient feature design, temporal modelling, and disciplined runtime engineering can together produce a system that is both technically credible and socially meaningful (Gan et al., 2023; Holmes et al., 2024; Tan et al., 2024). This is one of the most valuable lessons of the work. By demonstrating a realistic compromise between ambition and deployability, the project establishes a solid basis for future research, refinement, and eventual extension toward more natural and context-aware sign-language translation systems.

It therefore stands as a practical demonstration that accessibility-focused AI can be ambitious, technically grounded, and realistically deployable at the same time.

This overall outcome gives the project lasting academic and practical value.

## References

Alkhoraif, A. A., Alsulaiman, M., Abdul, W., & Bencherif, M. (2025). Ensemble transformer-based word-level sign language recognition with multi-modal input fusion. *Journal of Engineering Research*. Advance online publication. https://doi.org/10.1016/j.jer.2025.07.006

Alsharif, B., Alalwany, E., Ibrahim, A., Mahgoub, I., & Ilyas, M. (2025). Real-time American Sign Language interpretation using deep learning and keypoint tracking. *Sensors, 25*(7), Article 2138. https://doi.org/10.3390/s25072138

Anturkar, A., Khot, A., Andure, A., Ghosh, A., Magadum, A., Bahadur, A., & Pol, M. (2025). Real-time sign language to text translation using deep learning: A comparative study of LSTM and 3D CNN. *International Journal of Computer Applications, 187*(55). https://doi.org/10.5120/ijca2025925946

Badadhe, S. M., Sonsale, D., Jannawar, P., Somani, R., Verma, A., Tapar, S., & Kanakdande, P. (2025). Real-time American Sign Language to speech conversion using CNN and computer vision. *International Journal for Research in Applied Science and Engineering Technology, 13*(12), 66-73. https://doi.org/10.22214/ijraset.2025.76012

Camgoz, N. C., Koller, O., Hadfield, S., & Bowden, R. (2020). Sign language transformers: Joint end-to-end sign language recognition and translation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 10023-10033). https://openaccess.thecvf.com/content_CVPR_2020/papers/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.pdf

De Coster, M., Rushe, E., Holmes, R., Ventresque, A., & Dambre, J. (2023). Towards the extraction of robust sign embeddings for low resource sign language recognition. *arXiv preprint arXiv:2306.17558*. https://doi.org/10.48550/arXiv.2306.17558

Duarte, A., Palaskar, S., Ventura, L., Ghadiyaram, D., DeHaan, K., Metze, F., Torres, J., & Giro-i-Nieto, X. (2021). How2Sign: A large-scale multimodal dataset for continuous American Sign Language. *arXiv preprint arXiv:2008.08143*. https://arxiv.org/abs/2008.08143

Gan, S., Yin, Y., Jiang, Z., Xie, L., & Lu, S. (2023). Towards real-time sign language recognition and translation on edge devices. In *Proceedings of the ACM Web Conference 2023* (pp. 4502-4512). https://doi.org/10.1145/3581783.3611820

Holmes, R., Rushe, E., & Ventresque, A. (2024). The key points: Using feature importance to identify shortcomings in sign language recognition models. In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation* (pp. 15970-15980). https://aclanthology.org/2024.lrec-main.1387.pdf

Kalaiselvi, G., Badri, N., & Karnan, C. (2025). SignSpeak - Sign language translation system for hearing impaired. *International Journal of Science and Research Archive, 15*(2), 921-930. https://doi.org/10.30574/ijsra.2025.15.2.1523

Kamble, S. (2025). SLRNet: A real-time LSTM-based sign language recognition system. *arXiv preprint arXiv:2506.11154*. https://doi.org/10.48550/arXiv.2506.11154

Key, D. (2025). Real-time American Sign Language recognition using 3D convolutional neural networks and LSTM: Architecture, training, and deployment. *arXiv preprint arXiv:2512.22177*. https://doi.org/10.48550/arXiv.2512.22177

Kosna, S. R. (2025). *A real-time webcam-based system for sign language and speech translation* [ResearchGate preprint]. https://www.researchgate.net/publication/389660194_A_Real-Time_Webcam-Based_System_for_Sign_Language_and_Speech_Translation

Li, D., Opazo, C. R., Yu, X., & Li, H. (2020). Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. In *Proceedings of the IEEE Winter Conference on Applications of Computer Vision* (pp. 1459-1469). https://arxiv.org/abs/1910.11006

Naz, N., Sajid, H., Ali, S., Hasan, O., & Ehsan, M. K. (2023). MIPA-ResGCN: A multi-input part attention enhanced residual graph convolutional framework for sign language recognition. *Computers and Electrical Engineering, 112*, Article 109009. https://doi.org/10.1016/j.compeleceng.2023.109009

Tan, S., Khan, N., An, Z., Ando, Y., Kawakami, R., & Nakadai, K. (2024). A review of deep learning-based approaches to sign language processing. *Advanced Robotics, 38*, 1-19. https://doi.org/10.1080/01691864.2024.2442721

Uddin, M. Z., Boletsis, C., & Rudshavn, P. (2025). Real-time Norwegian Sign Language recognition using MediaPipe and LSTM. *Multimodal Technologies and Interaction, 9*(3), Article 23. https://doi.org/10.3390/mti9030023

Woods, L. T., & Rana, Z. A. (2023a). Modelling sign language with encoder-only transformers and human pose estimation keypoint data. *Mathematics, 11*(9), Article 2129. https://doi.org/10.3390/math11092129

Woods, L. T., & Rana, Z. A. (2023b). Constraints on optimising encoder-only transformers for modelling sign language with human pose estimation keypoint data. *Journal of Imaging, 9*(11), Article 238. https://doi.org/10.3390/jimaging9110238

## APPENDIX A. FULL HELD-OUT PER-CLASS METRICS

Table A.1 provides the complete per-class precision, recall, F1-score, and support values for the `300`-class held-out evaluation discussed in the Results section.

| Gloss | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ABOUT | 0.250 | 0.500 | 0.333 | 2 |
| ACCIDENT | 0.333 | 0.333 | 0.333 | 3 |
| AFRICA | 1.000 | 0.333 | 0.500 | 3 |
| AGAIN | 0.000 | 0.000 | 0.000 | 2 |
| ALL | 0.667 | 0.667 | 0.667 | 3 |
| ALWAYS | 1.000 | 1.000 | 1.000 | 2 |
| ANIMAL | 0.400 | 1.000 | 0.571 | 2 |
| APPLE | 1.000 | 1.000 | 1.000 | 2 |
| APPROVE | 0.000 | 0.000 | 0.000 | 2 |
| ARGUE | 1.000 | 1.000 | 1.000 | 2 |
| ARRIVE | 0.500 | 0.500 | 0.500 | 2 |
| BABY | 0.000 | 0.000 | 0.000 | 2 |
| BACK | 0.000 | 0.000 | 0.000 | 2 |
| BACKPACK | 0.000 | 0.000 | 0.000 | 3 |
| BAD | 0.000 | 0.000 | 0.000 | 2 |
| BAKE | 1.000 | 1.000 | 1.000 | 2 |
| BALANCE | 0.333 | 0.500 | 0.400 | 2 |
| BALL | 0.667 | 1.000 | 0.800 | 2 |
| BANANA | 0.400 | 1.000 | 0.571 | 2 |
| BAR | 0.000 | 0.000 | 0.000 | 3 |
| BASKETBALL | 1.000 | 0.500 | 0.667 | 2 |
| BATH | 1.000 | 0.500 | 0.667 | 2 |
| BATHROOM | 0.500 | 0.500 | 0.500 | 2 |
| BEARD | 1.000 | 1.000 | 1.000 | 2 |
| BECAUSE | 1.000 | 0.500 | 0.667 | 2 |
| BED | 1.000 | 1.000 | 1.000 | 2 |
| BEFORE | 0.000 | 0.000 | 0.000 | 4 |
| BEHIND | 0.333 | 0.500 | 0.400 | 2 |
| BIRD | 0.250 | 0.500 | 0.333 | 2 |
| BIRTHDAY | 0.000 | 0.000 | 0.000 | 3 |
| BLACK | 0.667 | 0.667 | 0.667 | 3 |
| BLANKET | 0.000 | 0.000 | 0.000 | 2 |
| BLUE | 0.667 | 1.000 | 0.800 | 2 |
| BOOK | 0.600 | 0.750 | 0.667 | 4 |
| BOWLING | 1.000 | 0.500 | 0.667 | 2 |
| BOY | 0.667 | 1.000 | 0.800 | 2 |
| BRING | 0.000 | 0.000 | 0.000 | 2 |
| BROTHER | 0.000 | 0.000 | 0.000 | 2 |
| BROWN | 1.000 | 0.500 | 0.667 | 2 |
| BUSINESS | 0.500 | 1.000 | 0.667 | 2 |
| BUT | 1.000 | 0.333 | 0.500 | 3 |
| BUY | 0.000 | 0.000 | 0.000 | 2 |
| CALL | 0.500 | 0.500 | 0.500 | 2 |
| CAN | 1.000 | 1.000 | 1.000 | 2 |
| CANDY | 1.000 | 0.333 | 0.500 | 3 |
| CAREFUL | 1.000 | 1.000 | 1.000 | 2 |
| CAT | 0.667 | 1.000 | 0.800 | 2 |
| CATCH | 0.000 | 0.000 | 0.000 | 2 |
| CENTER | 0.500 | 0.500 | 0.500 | 2 |
| CEREAL | 0.000 | 0.000 | 0.000 | 2 |
| CHAIR | 1.000 | 0.333 | 0.500 | 3 |
| CHAMPION | 0.500 | 0.500 | 0.500 | 2 |
| CHANGE | 1.000 | 0.500 | 0.667 | 2 |
| CHAT | 0.500 | 1.000 | 0.667 | 2 |
| CHEAT | 1.000 | 0.500 | 0.667 | 2 |
| CHECK | 0.000 | 0.000 | 0.000 | 3 |
| CHEESE | 0.500 | 0.500 | 0.500 | 2 |
| CHILDREN | 0.000 | 0.000 | 0.000 | 2 |
| CHRISTMAS | 1.000 | 0.500 | 0.667 | 2 |
| CITY | 0.400 | 1.000 | 0.571 | 2 |
| CLASS | 1.000 | 1.000 | 1.000 | 2 |
| CLOCK | 1.000 | 0.500 | 0.667 | 2 |
| CLOSE | 0.000 | 0.000 | 0.000 | 2 |
| CLOTHES | 0.333 | 0.333 | 0.333 | 3 |
| COFFEE | 0.500 | 0.500 | 0.500 | 2 |
| COLD | 0.500 | 1.000 | 0.667 | 2 |
| COLLEGE | 0.000 | 0.000 | 0.000 | 2 |
| COLOR | 0.500 | 0.500 | 0.500 | 2 |
| COMPUTER | 0.750 | 0.600 | 0.667 | 5 |
| CONVINCE | 1.000 | 0.500 | 0.667 | 2 |
| COOK | 0.000 | 0.000 | 0.000 | 2 |
| COOL | 0.333 | 0.333 | 0.333 | 3 |
| COPY | 1.000 | 0.500 | 0.667 | 2 |
| CORN | 0.000 | 0.000 | 0.000 | 3 |
| COUGH | 1.000 | 1.000 | 1.000 | 2 |
| COUNTRY | 0.500 | 1.000 | 0.667 | 2 |
| COUSIN | 1.000 | 0.667 | 0.800 | 3 |
| COW | 0.667 | 0.667 | 0.667 | 3 |
| CRASH | 1.000 | 0.500 | 0.667 | 2 |
| CRAZY | 0.000 | 0.000 | 0.000 | 2 |
| CRY | 0.400 | 1.000 | 0.571 | 2 |
| CUTE | 0.000 | 0.000 | 0.000 | 2 |
| DANCE | 0.500 | 0.500 | 0.500 | 2 |
| DARK | 0.500 | 0.333 | 0.400 | 3 |
| DAUGHTER | 0.250 | 0.500 | 0.333 | 2 |
| DAY | 1.000 | 1.000 | 1.000 | 2 |
| DEAF | 0.333 | 0.333 | 0.333 | 3 |
| DECIDE | 1.000 | 0.500 | 0.667 | 2 |
| DELAY | 0.333 | 0.500 | 0.400 | 2 |
| DELICIOUS | 0.000 | 0.000 | 0.000 | 2 |
| DIFFERENT | 0.500 | 1.000 | 0.667 | 2 |
| DISAPPEAR | 0.500 | 0.500 | 0.500 | 2 |
| DISCUSS | 0.667 | 1.000 | 0.800 | 2 |
| DIVORCE | 1.000 | 0.500 | 0.667 | 2 |
| DOCTOR | 1.000 | 0.333 | 0.500 | 3 |
| DOG | 0.500 | 0.667 | 0.571 | 3 |
| DOOR | 0.667 | 1.000 | 0.800 | 2 |
| DRAW | 1.000 | 0.500 | 0.667 | 2 |
| DRESS | 0.000 | 0.000 | 0.000 | 2 |
| DRINK | 0.500 | 0.250 | 0.333 | 4 |
| DRIVE | 1.000 | 0.500 | 0.667 | 2 |
| DROP | 1.000 | 1.000 | 1.000 | 2 |
| EAST | 0.500 | 0.500 | 0.500 | 2 |
| EASY | 0.500 | 1.000 | 0.667 | 2 |
| EAT | 1.000 | 0.500 | 0.667 | 2 |
| EGG | 0.333 | 0.500 | 0.400 | 2 |
| ENJOY | 0.000 | 0.000 | 0.000 | 2 |
| ENVIRONMENT | 0.000 | 0.000 | 0.000 | 2 |
| EXAMPLE | 1.000 | 0.500 | 0.667 | 2 |
| FAMILY | 1.000 | 1.000 | 1.000 | 2 |
| FAR | 0.000 | 0.000 | 0.000 | 3 |
| FAT | 1.000 | 0.500 | 0.667 | 2 |
| FATHER | 0.000 | 0.000 | 0.000 | 2 |
| FAULT | 1.000 | 0.500 | 0.667 | 2 |
| FEEL | 1.000 | 0.500 | 0.667 | 2 |
| FINE | 0.000 | 0.000 | 0.000 | 3 |
| FINISH | 0.667 | 0.667 | 0.667 | 3 |
| FIRST | 0.500 | 0.500 | 0.500 | 2 |
| FISH | 1.000 | 0.667 | 0.800 | 3 |
| FLOWER | 1.000 | 1.000 | 1.000 | 2 |
| FOOTBALL | 0.667 | 1.000 | 0.800 | 2 |
| FORGET | 0.000 | 0.000 | 0.000 | 2 |
| FRIEND | 1.000 | 0.500 | 0.667 | 2 |
| FRIENDLY | 0.000 | 0.000 | 0.000 | 2 |
| FULL | 1.000 | 0.500 | 0.667 | 2 |
| FUTURE | 1.000 | 1.000 | 1.000 | 2 |
| GAME | 0.333 | 1.000 | 0.500 | 2 |
| GIRL | 0.250 | 0.500 | 0.333 | 2 |
| GIVE | 0.667 | 1.000 | 0.800 | 2 |
| GLASSES | 0.667 | 1.000 | 0.800 | 2 |
| GO | 0.667 | 0.667 | 0.667 | 3 |
| GOOD | 0.000 | 0.000 | 0.000 | 2 |
| GOVERNMENT | 0.400 | 1.000 | 0.571 | 2 |
| GRADUATE | 0.500 | 1.000 | 0.667 | 2 |
| GREEN | 0.000 | 0.000 | 0.000 | 2 |
| HAIR | 1.000 | 0.500 | 0.667 | 2 |
| HALLOWEEN | 0.333 | 0.500 | 0.400 | 2 |
| HAPPY | 0.000 | 0.000 | 0.000 | 2 |
| HARD | 0.500 | 0.500 | 0.500 | 2 |
| HAT | 1.000 | 0.500 | 0.667 | 2 |
| HAVE | 1.000 | 0.500 | 0.667 | 2 |
| HEADACHE | 0.667 | 1.000 | 0.800 | 2 |
| HEAR | 1.000 | 0.500 | 0.667 | 2 |
| HEARING | 1.000 | 0.500 | 0.667 | 2 |
| HEART | 1.000 | 1.000 | 1.000 | 2 |
| HELP | 1.000 | 1.000 | 1.000 | 3 |
| HERE | 0.000 | 0.000 | 0.000 | 2 |
| HOME | 0.500 | 0.500 | 0.500 | 2 |
| HOPE | 1.000 | 0.500 | 0.667 | 2 |
| HOT | 1.000 | 0.667 | 0.800 | 3 |
| HOUR | 0.000 | 0.000 | 0.000 | 2 |
| HOUSE | 0.500 | 0.500 | 0.500 | 2 |
| HOW | 0.750 | 1.000 | 0.857 | 3 |
| HUMBLE | 0.250 | 0.500 | 0.333 | 2 |
| HURRY | 0.500 | 1.000 | 0.667 | 2 |
| HUSBAND | 1.000 | 0.500 | 0.667 | 2 |
| IMPROVE | 0.333 | 0.500 | 0.400 | 2 |
| INFORM | 1.000 | 0.667 | 0.800 | 3 |
| INTEREST | 0.000 | 0.000 | 0.000 | 2 |
| INTERNET | 1.000 | 1.000 | 1.000 | 2 |
| JACKET | 0.000 | 0.000 | 0.000 | 2 |
| JOIN | 1.000 | 0.500 | 0.667 | 2 |
| JUMP | 0.400 | 1.000 | 0.571 | 2 |
| KILL | 0.000 | 0.000 | 0.000 | 2 |
| KISS | 0.250 | 0.333 | 0.286 | 3 |
| KNIFE | 0.500 | 0.500 | 0.500 | 2 |
| KNOW | 0.000 | 0.000 | 0.000 | 2 |
| LANGUAGE | 1.000 | 0.333 | 0.500 | 3 |
| LAST | 0.667 | 0.667 | 0.667 | 3 |
| LATE | 1.000 | 1.000 | 1.000 | 2 |
| LATER | 0.000 | 0.000 | 0.000 | 2 |
| LAUGH | 0.500 | 0.333 | 0.400 | 3 |
| LAW | 0.500 | 1.000 | 0.667 | 2 |
| LEARN | 0.400 | 1.000 | 0.571 | 2 |
| LEAVE | 0.000 | 0.000 | 0.000 | 2 |
| LETTER | 0.250 | 0.333 | 0.286 | 3 |
| LIGHT | 1.000 | 0.500 | 0.667 | 2 |
| LIKE | 1.000 | 0.667 | 0.800 | 3 |
| LIST | 0.000 | 0.000 | 0.000 | 2 |
| LIVE | 1.000 | 0.500 | 0.667 | 2 |
| LOSE | 0.000 | 0.000 | 0.000 | 2 |
| MAKE | 0.667 | 1.000 | 0.800 | 2 |
| MAN | 0.167 | 0.500 | 0.250 | 2 |
| MANY | 0.750 | 1.000 | 0.857 | 3 |
| MATCH | 1.000 | 0.500 | 0.667 | 2 |
| MEAN | 0.000 | 0.000 | 0.000 | 2 |
| MEAT | 1.000 | 0.500 | 0.667 | 2 |
| MEDICINE | 0.500 | 0.500 | 0.500 | 2 |
| MEET | 1.000 | 1.000 | 1.000 | 2 |
| MILK | 0.500 | 0.500 | 0.500 | 2 |
| MONEY | 0.667 | 1.000 | 0.800 | 2 |
| MORE | 0.250 | 0.500 | 0.333 | 2 |
| MOST | 0.400 | 1.000 | 0.571 | 2 |
| MOTHER | 0.333 | 0.333 | 0.333 | 3 |
| MOVIE | 0.000 | 0.000 | 0.000 | 2 |
| MUSIC | 0.000 | 0.000 | 0.000 | 2 |
| NAME | 0.250 | 0.500 | 0.333 | 2 |
| NEED | 1.000 | 0.500 | 0.667 | 2 |
| NEW | 0.400 | 1.000 | 0.571 | 2 |
| NO | 1.000 | 1.000 | 1.000 | 3 |
| NONE | 1.000 | 1.000 | 1.000 | 2 |
| NOW | 0.500 | 0.667 | 0.571 | 3 |
| OFFICE | 0.667 | 1.000 | 0.800 | 2 |
| OLD | 1.000 | 0.500 | 0.667 | 2 |
| ORANGE | 0.750 | 1.000 | 0.857 | 3 |
| ORDER | 0.667 | 1.000 | 0.800 | 2 |
| PAINT | 1.000 | 0.500 | 0.667 | 2 |
| PANTS | 0.333 | 0.500 | 0.400 | 2 |
| PAPER | 0.000 | 0.000 | 0.000 | 2 |
| PARTY | 0.000 | 0.000 | 0.000 | 2 |
| PAST | 0.000 | 0.000 | 0.000 | 2 |
| PENCIL | 0.667 | 1.000 | 0.800 | 2 |
| PERSON | 0.000 | 0.000 | 0.000 | 2 |
| PINK | 0.000 | 0.000 | 0.000 | 3 |
| PIZZA | 1.000 | 0.333 | 0.500 | 3 |
| PLAN | 0.333 | 0.500 | 0.400 | 2 |
| PLAY | 0.250 | 0.500 | 0.333 | 2 |
| PLEASE | 0.400 | 1.000 | 0.571 | 2 |
| POLICE | 0.667 | 1.000 | 0.800 | 2 |
| PRACTICE | 0.333 | 0.500 | 0.400 | 2 |
| PRESIDENT | 0.500 | 1.000 | 0.667 | 2 |
| PROBLEM | 1.000 | 1.000 | 1.000 | 2 |
| PULL | 0.667 | 1.000 | 0.800 | 2 |
| PURPLE | 0.000 | 0.000 | 0.000 | 2 |
| RABBIT | 1.000 | 0.333 | 0.500 | 3 |
| READ | 0.500 | 1.000 | 0.667 | 2 |
| RED | 1.000 | 0.500 | 0.667 | 2 |
| REMEMBER | 0.000 | 0.000 | 0.000 | 2 |
| RESTAURANT | 0.333 | 0.500 | 0.400 | 2 |
| RIDE | 0.500 | 0.500 | 0.500 | 2 |
| RIGHT | 0.000 | 0.000 | 0.000 | 2 |
| ROOM | 1.000 | 0.333 | 0.500 | 3 |
| RUN | 0.500 | 0.500 | 0.500 | 2 |
| RUSSIA | 0.250 | 0.500 | 0.333 | 2 |
| SALT | 0.000 | 0.000 | 0.000 | 2 |
| SAME | 0.500 | 0.667 | 0.571 | 3 |
| SANDWICH | 0.250 | 0.500 | 0.333 | 2 |
| SCHOOL | 0.000 | 0.000 | 0.000 | 2 |
| SECRETARY | 1.000 | 0.333 | 0.500 | 3 |
| SHARE | 0.200 | 0.500 | 0.286 | 2 |
| SHIRT | 1.000 | 0.667 | 0.800 | 3 |
| SHORT | 0.500 | 0.500 | 0.500 | 2 |
| SHOW | 1.000 | 1.000 | 1.000 | 2 |
| SICK | 0.500 | 0.500 | 0.500 | 2 |
| SIGN | 0.500 | 0.500 | 0.500 | 2 |
| SINCE | 1.000 | 1.000 | 1.000 | 2 |
| SMALL | 0.000 | 0.000 | 0.000 | 2 |
| SNOW | 0.500 | 1.000 | 0.667 | 2 |
| SOME | 0.000 | 0.000 | 0.000 | 2 |
| SON | 0.000 | 0.000 | 0.000 | 2 |
| SOON | 0.400 | 0.667 | 0.500 | 3 |
| SOUTH | 0.000 | 0.000 | 0.000 | 2 |
| STAY | 0.500 | 0.500 | 0.500 | 2 |
| STUDENT | 1.000 | 0.500 | 0.667 | 2 |
| STUDY | 0.667 | 1.000 | 0.800 | 2 |
| SUNDAY | 1.000 | 1.000 | 1.000 | 2 |
| TABLE | 1.000 | 1.000 | 1.000 | 3 |
| TAKE | 0.333 | 0.333 | 0.333 | 3 |
| TALL | 0.333 | 0.333 | 0.333 | 3 |
| TEA | 1.000 | 1.000 | 1.000 | 2 |
| TEACH | 0.667 | 1.000 | 0.800 | 2 |
| TEACHER | 0.667 | 1.000 | 0.800 | 2 |
| TELL | 1.000 | 0.500 | 0.667 | 2 |
| TEST | 1.000 | 1.000 | 1.000 | 2 |
| THANKSGIVING | 0.500 | 0.333 | 0.400 | 3 |
| THEORY | 0.333 | 0.500 | 0.400 | 2 |
| THIN | 0.000 | 0.000 | 0.000 | 3 |
| THURSDAY | 1.000 | 1.000 | 1.000 | 2 |
| TIME | 0.667 | 1.000 | 0.800 | 2 |
| TIRED | 1.000 | 0.500 | 0.667 | 2 |
| TOMATO | 0.500 | 1.000 | 0.667 | 2 |
| TRADE | 0.000 | 0.000 | 0.000 | 2 |
| TRAIN | 0.200 | 0.500 | 0.286 | 2 |
| TRAVEL | 0.667 | 1.000 | 0.800 | 2 |
| UGLY | 0.500 | 0.500 | 0.500 | 2 |
| VISIT | 1.000 | 1.000 | 1.000 | 2 |
| WAIT | 0.500 | 0.500 | 0.500 | 2 |
| WALK | 0.333 | 0.333 | 0.333 | 3 |
| WANT | 0.200 | 0.500 | 0.286 | 2 |
| WAR | 0.000 | 0.000 | 0.000 | 2 |
| WATER | 0.500 | 1.000 | 0.667 | 2 |
| WEEK | 1.000 | 0.500 | 0.667 | 2 |
| WHAT | 0.333 | 0.333 | 0.333 | 3 |
| WHERE | 0.667 | 1.000 | 0.800 | 2 |
| WHITE | 1.000 | 1.000 | 1.000 | 2 |
| WHO | 0.000 | 0.000 | 0.000 | 3 |
| WHY | 0.400 | 1.000 | 0.571 | 2 |
| WIFE | 0.333 | 0.500 | 0.400 | 2 |
| WINDOW | 0.500 | 0.500 | 0.500 | 2 |
| WITH | 1.000 | 0.500 | 0.667 | 2 |
| WOMAN | 1.000 | 0.667 | 0.800 | 3 |
| WORK | 0.500 | 0.500 | 0.500 | 2 |
| WRITE | 0.333 | 0.500 | 0.400 | 2 |
| WRONG | 0.667 | 1.000 | 0.800 | 2 |
| YEAR | 0.600 | 1.000 | 0.750 | 3 |
| YELLOW | 0.500 | 0.500 | 0.500 | 2 |
| YES | 0.200 | 0.333 | 0.250 | 3 |
| YESTERDAY | 0.000 | 0.000 | 0.000 | 2 |
| YOU | 0.000 | 0.000 | 0.000 | 2 |
| YOUR | 1.000 | 1.000 | 1.000 | 2 |

## APPENDIX B. CLEANED LIVE-TESTING SHEET

The pasted live-testing worksheet contained many routine `x -> s` rows with no unusual notes. For readability, Table B.1 cleans and condenses the entries that carried explicit observations, repeated confusions, unresolved behaviour, or noteworthy refinement outcomes. The aggregate live results remained `82.7%` initially and improved to `91.67%` after refinement.

| Gloss | Initial live status | Refined/final status | Cleaned observation |
|---|---|---|---|
| AGAIN | `f` | unresolved | Did not appear in the Top-5 list during the noted attempt. |
| APPROVE | `f2` | unresolved | Repeatedly appeared in Top-5 and was detected at least once, but was commonly confused with `DRAW`. |
| ARRIVE | `f2` | unresolved | Spiked to roughly `70-80%`, then decayed and was replaced by `BABY` at the end of the gloss. |
| BAD | `f2` | unresolved | Inconsistent identification; often ended as `SCHOOL`. |
| BEFORE | `f2` | unresolved | Appeared in Top-5 but did not finalize; often confused with `CLOSE` and `WINDOW`. |
| BEHIND | `f2` | unresolved | Appeared in Top-5 but was often confused with `WITH`. |
| BOOK | `f2` | resolved | Difficult to finalize, but repeatedly appeared in Top-5. |
| BRING | `f` | resolved | Initially absent from Top-5. |
| BUT | pending | pending | Needed a larger outward pull toward shoulder position. |
| CALL | `f` | resolved | Completely undetected in the noted live attempt. |
| CATCH | `f2` | unresolved | Brief `40-50%` spike, then changed to `YEAR` at the end pose. |
| CENTER | `f2` | unresolved | Detectable in Top-5 but repeatedly mistaken for `DOCTOR`. |
| CEREAL | pending | pending | Could trigger even when the hand was stationary; required stricter motion handling. |
| CHANGE | `f2` | resolved | Appeared in Top-5 but did not finalize. |
| CHAT | `f` | resolved | Initially absent from Top-5. |
| CHEESE | `f2` | unresolved | Appeared in Top-5 but still collapsed into `SCHOOL`. |
| CHILDREN | `f2` | unresolved | Commonly mistaken for `SHORT`. |
| CHRISTMAS | `f2` | unresolved | Briefly spiked to about `50%` but still did not finalize. |
| COLLEGE | `f2` | resolved | Difficult to detect, but repeatedly appeared in Top-5. |
| COOK | `f2` | resolved | Appeared in Top-5 but required refinement. |
| COPY | `f2` | resolved | Appeared in Top-5 but required refinement. |
| CORN | `f2` | unresolved | Commonly mistaken for `SIGN` and `CHAIR`. |
| CRASH | `f2` | unresolved | Mistaken for `ACCIDENT`, and later for `PROBLEM` from the static end pose. |
| CRAZY | `f2` | unresolved | Continued to be mistaken for `COUSIN`. |
| DAUGHTER | inconsistent | resolved | Identified inconsistently before refinement. |
| DOCTOR | pending | pending | Confused with `AGAIN` and could also trigger from an incorrect static pose; needed explicit motion-only handling. |
| DRESS | `f2` | resolved | Appeared in Top-5 but did not finalize at first. |
| EASY | `f2` | resolved | Continued to overlap with `CEREAL` despite being a different sign. |
| EXAMPLE | `f2` | unresolved | Still mistaken for `SHOW`. |
| FAR | `f2` | unresolved | Still mistaken for `GAME`. |
| FEEL | `f2` | resolved | Improved when the upward flicking motion was repeated more clearly. |
| FINE | pending | pending | Reached Top-1 around `50-60%` but often stayed below the speaking threshold. |
| FINISH | `f2` | resolved | Needed a stronger push-like completion. |
| FIRST | `f2` | unresolved | Mistaken for `YEAR` or `PROBLEM`. |
| GO | `f` | resolved | Initially absent from Top-5. |
| HARD | `f` | resolved | Often mistaken for `PROBLEM`. |
| HEARING | pending | pending | Could trigger while stationary or moving sideways instead of only on the intended up-down motion. |
| HOPE | `f2` | unresolved | Often confused with `WAR`; could spike to `60-90%` only briefly. |
| INTEREST | pending | pending | Inconsistently identified and sometimes mistaken for `TABLE`. |
| JACKET | pending | pending | Could spike to `90%` briefly, then switch to `BATH`. |
| KISS | `f2` | resolved in Top-5 only | Repeatedly appeared in Top-5 but consistently collapsed into `MORE`. |
| LAW | `f2` | unresolved | Confused with `HOUR`; the correct `LAW` spike often dropped after the gloss ended. |
| MEAN | `f2` | resolved | Improved when ending more clearly with thumbs-up. |
| MUSIC | `f2` | unresolved | Appeared but did not finalize in the noted attempt. |
| ORDER | `f2` | resolved | Appeared in Top-5 but required refinement. |
| PAPER | `f` | unresolved | Did not appear in Top-5 in the noted attempt. |
| PAST | inconsistent | resolved | Initially overlapped with `LAST`, then improved. |
| PENCIL | inconsistent | resolved | Initially overlapped with `WRITE`, then improved. |
| RABBIT | `f2` | unresolved | Remained difficult to raise beyond `50%`. |
| SAME | `f2` | resolved | Model confused `SAME` and `STAY` during attempts to sign `SAME`. |
| SECRETARY | `f2` | unresolved | Continued to be mistaken for `PENCIL`. |
| SIGN | `f2` | unresolved | Could rank first, but often remained below `50%` and therefore was not spoken. |
| TEST | pending | pending | Could trigger from one or both stationary index fingers, causing overlap with `DIFFERENT` or `BUT`. |
| WHY | `f2` | unresolved | Appeared in Top-5 but did not finalize in the noted sheet entry. |

Table B.2 summarizes representative live notes for successful but instruction-sensitive signs.

| Gloss | Live outcome | Cleaned execution note |
|---|---|---|
| ACCIDENT | success | Needed clearer curling of the index and middle fingers. |
| ALL | success | More reliable when performed farther from the camera; right hand needed a clearer swoosh motion. |
| ANIMAL | success | More reliable when the hands curved inward clearly. |
| APPLE | success | Needed the right hand to be used clearly. |
| BACKPACK | success | More reliable when signed from top to down on the shoulders to avoid overlap with `ANIMAL`. |
| BATHROOM | success | More reliable when the wrist moved and the rest of the arm stayed still. |
| BECAUSE | success | More reliable when the swipe stayed close to the forehead. |
| BROTHER | success | Needed a forehead start followed by a clearer finger-gun transition. |
| CANDY | success | Could still overlap with `CEREAL`, but the cheek-point position improved reliability. |
| CAREFUL | success | More reliable when the top hand clearly lowered onto the lower hand. |
| CHAIR | success | More reliable when signed from the side. |
| CLASS | success | More reliable with a clear open-claw rotation. |
| COMPUTER | success | Needed clearer cup-hand orientation and arm-relative circling. |
| CONVINCE | success | Depended on full detection of the chopping hand. |
| DAY | success | More reliable when signed more slowly. |
| DEAF | success | Needed a clear mouth-to-ear path with the finger at roughly `90` degrees. |
| DELICIOUS | success | More reliable when signed from the side. |
| DISAPPEAR | success | Improved when signed more slowly. |
| GOVERNMENT | success | More reliable when the wrist moved while the hand stayed otherwise stable. |
| LIST | success | Improved when the body was tilted slightly left. |
| MAN | success | Required the elbow to stay visible. |
| PAINT | success | Needed the left open palm tilted toward the camera. |
| SCHOOL | success | Needed a clearer two-hand clapping motion instead of a static base hand. |
| SNOW | success | More reliable when started at head level with flatter palms. |
| WIFE | success | Needed shoulder-level start and a stationary left hand before motion. |
| YOUR | success | More reliable when the palm moved toward the camera. |
