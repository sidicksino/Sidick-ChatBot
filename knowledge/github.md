# My GitHub repositories

## GitHub project: Sidick-ChatBot
This is one of my GitHub repositories: Sidick-ChatBot.
Main language: Python.
Code: https://github.com/sidicksino/Sidick-ChatBot (updated 2026-11-19, 0 stars).
README:
Sidick-ChatBot

## GitHub project: ChatBot-Elon
This is one of my GitHub repositories: ChatBot-Elon.
Main language: Jupyter Notebook.
Code: https://github.com/sidicksino/ChatBot-Elon (updated 2026-11-19, 0 stars).
README:
ChatBot-Elon

## GitHub project: sidicksino
This is one of my GitHub repositories: sidicksino.
Code: https://github.com/sidicksino/sidicksino (updated 2026-11-19, 0 stars).
README:
Portfolio
  &nbsp;&middot;&nbsp;
  LinkedIn
  &nbsp;&middot;&nbsp;
  X
  &nbsp;&middot;&nbsp;
  Email

I build web platforms, mobile apps and machine-learning tools, taking an idea from raw data and a rough design through to something people can actually use. I'm a Data Science student working between N'Djamena, Chad and Kigali, Rwanda.

My goal is to build digital products that genuinely serve students and entrepreneurs across Africa.

Selected work

  
    
      
      AAPT
      Web platform for Association Actions Pour Tous Tchad, a non-profit working on social cohesion and community support.
      React · Tailwind CSS · Supabase · i18next
      Live site
    
    
      
      Sino Ai
      AI assistant web app with saved conversations, built on a Node.js and MongoDB backend.
      React · Node.js · MongoDB · OpenAI API
      Live site &nbsp;·&nbsp; Code
    
  
  
    
      
      Disease Risk Prediction
      Machine-learning app that estimates disease risk from a person's health and lifestyle inputs.
      Python · Scikit-learn · Pandas · Streamlit
      Live app &nbsp;·&nbsp; Code
    
    
      
      TchadInfos
      Mobile news app for Chad, with stories browsable by category and a live news feed.
      React Native · Expo · TypeScript · NewsData.io
      Code
    
  
  
    
      
      Académie Royale
      Official website for an educational institution in N'Djamena.
      React · Tailwind CSS · Supabase · i18next
      Live site
    
    
      
      SinoCoffee
      Coffee brand site with a menu, subscriptions, a cart and user sign-in.
      React · TypeScript · Supabase · Tailwind CSS
      Live site &nbsp;·&nbsp; Code
    
  

See all projects on my portfolio

Stack

**Data science**

Also Pandas, NumPy, Matplotlib, Seaborn, Power BI and Tableau

**Frontend**

**Mobile and design**

React Native, Expo and Figma

**Backend**

Activity

## GitHub project: Remotion-video
This is one of my GitHub repositories: Remotion-video.
Main language: HTML.
Code: https://github.com/sidicksino/Remotion-video (updated 2026-11-18, 0 stars).
README:
Remotion-video

A workspace for motion design and short-form video editing with Claude Code, using Remotion, HyperFrames, FFmpeg, Whisper and segmentation tools.

Setup
```bash
cd remotion-studio && npm install && npm run dev      # Remotion Studio
cd hyperframes-studio && npm install && npm run dev   # HyperFrames preview
brew install ffmpeg yt-dlp uv imagemagick whisper-cpp
cd tools && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python faster-whisper "rembg[cpu]" mediapipe opencv-python scenedetect auto-editor moviepy pillow numpy
```

See CLAUDE.md for the workspace guide, INSPIRATION.md for idea sources, and same-example/STYLE-NOTES.md for the reference style.

## GitHub project: DIY-AI-VIDEO-SKILL
This is one of my GitHub repositories: DIY-AI-VIDEO-SKILL.
Main language: Python.
Code: https://github.com/sidicksino/DIY-AI-VIDEO-SKILL (updated 2026-11-16, 0 stars).
README:
Luna Upcycles

AI-generated garden upcycling Shorts for YouTube and TikTok. Luna turns junk
into concrete garden pieces; every video ends on an animal enjoying it.

Layout

```
brand/      channel.md (setup + copy), luna.md (character prompts),
            realism-playbook.md, logo, watermark, banner, avatar
ideas/      log.md (every idea ever, never repeat), shortlist-*.md, ideas.md
projects/   one folder per video: plan.md, edit.json, frames/, clips/, output/
tools/      assemble.py (edit), review.py (frame-by-frame check), flow_frames.py (frames in Flow),
            analyze_ref.py (split a reference video into shots), branding.py, ...
music/      YouTube Audio Library tracks only
same-idea/  videos to remake with /same-idea (git-ignored)
references/ Instagram reference compilations (git-ignored)
```

Skills

- **/luna-ideas** (`.claude/skills/luna-ideas/`) — researches breakout DIY
  Shorts on vidIQ and saves a ranked shortlist of original Luna ideas. Ideas only.
- **/luna-plan** (`.claude/skills/luna-plan/`) — turns a picked idea into
  projects/NNN-slug/plan.md (hero + stage images, one-action clip prompts) and
  edit.json. Planning only.
- **/luna-generate** (`.claude/skills/luna-generate/`) — runs plan.md in Google
  Flow via flow-auto: hero (approve) → stage frames (approve) → Veo clips,
  each checked and saved into the project. Free plan: keep the Flow tab visible
  and the Flow Connector side panel open.

- **/same-idea** (`.claude/skills/same-idea/`) — remake a DIY video you liked:
  drop it in `same-idea/` (optional `.txt` with your changes), run
  /same-idea. It splits the video into shots (`tools/analyze_ref.py`) and writes a
  Luna project with the same shots and pacing, our character, place and design.
- **/luna-steps** (`.claude/skills/luna-steps/`) — turns a numbered
  step-by-step DIY image (drop it in `frames-images/`) into a Luna project of 4–6
  full 8 s clips, one action each, written second by second.
- **/luna-score** (`.claude/skills/luna-score/`) — scores a finished video
  against the researched growth rules in `brand/growth-playbook.md`
  (`tools/score_video.py`) and lists what to fix before posting.

Making a video

1. Pick an idea from `ideas/ideas.md`; its prompts go in
   `projects//prompts.md`.
2. In Google Flow, for each clip: start image with **Nano Banana 2** (x1,
   9:16, Luna character), then **Veo 3.1 Fast** (x1, 9:16, 8 s) from that
   image. Download to `projects//clips/c1.mp4`, `c2.mp4`, …
3. Adjust the trims in

## GitHub project: SinoTicket
This is one of my GitHub repositories: SinoTicket.
Main language: TypeScript.
Live demo: https://sino-ticket.vercel.app
Code: https://github.com/sidicksino/SinoTicket (updated 2026-11-16, 0 stars).
README:
SinoTicket

**Event ticketing for Chad: discover events, pick your seat, and keep your ticket on your phone.**

  
  &nbsp;
  
  &nbsp;
  

---

About

Paper tickets sold at the door mean queues on the day, no way to choose a seat in advance, and tickets that are easy to fake. Organizers also can't see how sales are going until the event starts.

**SinoTicket** puts the whole process on a phone:

- **Attendees** browse upcoming events, choose a ticket category and an exact seat, pay, and get a QR-code ticket in their wallet.
- **Organizers** create venues, sections, seats and events from a web dashboard, follow sales live, and scan tickets at the entrance with the same app.

The app is available in **English and French** and supports **light and dark mode**.

Features

For attendees
- **Discover events:** search by name and filter by category (Music, Sports, Cultural, Business, Fashion).
- **Event pages:** artwork, date and time, venue, line-up, and ticket prices in XAF.
- **Live seat map:** seats turn red on every phone the moment someone else holds or buys them.
- **Fair seat holds:** a chosen seat is held for 15 minutes during checkout, then released automatically if unpaid. Two people can never buy the same seat.
- **Checkout:** Mobile Money or card. Payment is simulated for now, so no real money moves.
- **Digital wallet:** each ticket has a unique QR code and can be downloaded as an image.
- **Notifications:** an in-app inbox plus push notifications when a booking is confirmed.
- **Accounts:** email or Google sign-in, profile photo, phone number, language, and theme.

For organizers (admins)
- **Ticket scanner:** scan a QR code at the door to validate the ticket and mark it as used, so it can't be used twice.
- **Admin dashboard (web):**
  - Revenue, tickets sold, users and events at a glance, with a six-month sales chart.
  - Manage venues, sections (nested), and seats (generate thousands at once).
  - Create and edit events: images, categories, line-up, and ticket categories linked to sections.
  - Browse all tickets and users.

Screenshots

| Home | Event | Tickets |
|:---:|:---:|:---:|
|  |  |  |
| **Seat map** | **Checkout** | **Wallet & QR code** |
|  |  |  |
| **Notifications** | **Dark mode** | |
|  |  | |

Tech stack

| Part | Technology |
|---|---|
| Mobile app | Expo (React Native), Expo Router, TypeScript, NativeWind, Reanimated |
| Backend | Convex: database, server functions, scheduled jobs, file storage |
| Authentication | Clerk: emai

## GitHub project: Mahamat-project-shop
This is one of my GitHub repositories: Mahamat-project-shop.
Main language: PHP.
Code: https://github.com/sidicksino/Mahamat-project-shop (updated 2026-10-30, 0 stars).
README:
FocalPoint: Camera and Photography Store

## GitHub project: Adam-Chaibo-Project-INSTA-forum
This is one of my GitHub repositories: Adam-Chaibo-Project-INSTA-forum.
Main language: JavaScript.
Code: https://github.com/sidicksino/Adam-Chaibo-Project-INSTA-forum (updated 2026-10-29, 0 stars).
README:
INSTA-FORUM

Forum de questions-réponses pour les étudiants, enseignants et le personnel du département de génie informatique de l’INSTA.

On y pose ses questions, on répond à celles des autres, on vote pour les meilleures réponses et l’auteur d’une question choisit celle qui a résolu son problème.

Fonctionnalités

- **Comptes** : inscription (prénom, nom, profil étudiant / enseignant / personnel, niveau), connexion avec le nom d’utilisateur **ou** l’email, option « Se souvenir de moi », déconnexion.
- **Questions** : publication avec catégorie et tags, recherche, onglets *Récentes*, *Populaires*, *Sans réponse*, *Résolues*.
- **Réponses** : page de détail d’une question, réponses, compteur de vues (une vue par membre).
- **Modifier / supprimer** : chacun peut corriger ou supprimer ses propres questions et réponses (mention « modifiée »).
- **Profil des membres** : cliquer sur un nom ouvre son profil (questions, réponses, statistiques, rang au classement). Sur son propre profil, l’onglet « Paramètres » permet de modifier ses informations et son mot de passe.
- **Mot de passe oublié** : lien envoyé par email (valable 1 heure, utilisable une seule fois) pour choisir un nouveau mot de passe.
- **Notifications** (🔔) : prévient quand quelqu’un répond à votre question ou accepte votre réponse ; pastille rouge, vérification toutes les 30 s, « Tout marquer comme lu ».
- **Captures d’écran** : jusqu’à 5 images par question (5 Mo max). Elles sont redimensionnées (1600 px) et converties en WebP à l’envoi, avec une miniature de 400 px affichée sur les cartes de l’accueil.
- **Code mis en forme** : le code entouré de ``` s’affiche en couleur (Python, Java, C, C++, C#, JavaScript, TypeScript, PHP, SQL, HTML, CSS, JSON, terminal), avec un bouton « Copier » ; `code` dans une phrase est aussi mis en valeur. Le bouton «  Code » des formulaires insère le bloc.
- **Réponse acceptée** : l’auteur de la question choisit la bonne réponse, la question passe en *Résolue*.
- **Votes** : pour (▲) ou contre (▼) une question ou une réponse ; impossible de voter pour soi-même.
- **Menu** : *Accueil*, *Questions* (filtre par catégorie), *Catégories*, *Mes questions*, *Favoris* (☆), *Classement*.
- **Classement** des membres par points (voir Barème).
- **Administration** Django pour gérer les utilisateurs, catégories, questions et réponses.

Technologies

| Partie | Outils |
|---|---|
| Frontend | React 18, Vite 5, highlight.js (coloration du code) |
| Backend | Django 6.1, Django REST F

## GitHub project: Restaurant-Analysis
This is one of my GitHub repositories: Restaurant-Analysis.
Main language: Python.
Code: https://github.com/sidicksino/Restaurant-Analysis (updated 2026-10-22, 0 stars).
README:
Restaurant Analysis

Computer vision and analytics pipeline for restaurant CCTV footage using YOLO object detection, table occupancy tracking, and spatial analytics.

---

COCO Dataset Classes Reference (YOLO)

Pre-trained YOLO models (such as YOLOv8 and YOLO11 trained on the standard **COCO** dataset) support **80 classes** indexed from `0` to `79`.

Full Class List (0 – 79)

| Class ID | Class Name | Class ID | Class Name |
| :---: | :--- | :---: | :--- |
| **0** | person | **40** | wine glass |
| **1** | bicycle | **41** | cup |
| **2** | car | **42** | fork |
| **3** | motorcycle | **43** | knife |
| **4** | airplane | **44** | spoon |
| **5** | bus | **45** | bowl |
| **6** | train | **46** | banana |
| **7** | truck | **47** | apple |
| **8** | boat | **48** | sandwich |
| **9** | traffic light | **49** | orange |
| **10** | fire hydrant | **50** | broccoli |
| **11** | stop sign | **51** | carrot |
| **12** | parking meter | **52** | hot dog |
| **13** | bench | **53** | pizza |
| **14** | bird | **54** | donut |
| **15** | cat | **55** | cake |
| **16** | dog | **56** | chair |
| **17** | horse | **57** | couch |
| **18** | sheep | **58** | potted plant |
| **19** | cow | **59** | bed |
| **20** | elephant | **60** | dining table |
| **21** | bear | **61** | toilet |
| **22** | zebra | **62** | tv |
| **23** | giraffe | **63** | laptop |
| **24** | backpack | **64** | mouse |
| **25** | umbrella | **65** | remote |
| **26** | handbag | **66** | keyboard |
| **27** | tie | **67** | cell phone |
| **28** | suitcase | **68** | microwave |
| **29** | frisbee | **69** | oven |
| **30** | skis | **70** | toaster |
| **31** | snowboard | **71** | sink |
| **32** | sports ball | **72** | refrigerator |
| **33** | kite | **73** | book |
| **34** | baseball bat | **74** | clock |
| **35** | baseball glove | **75** | vase |
| **36** | skateboard | **76** | scissors |
| **37** | surfboard | **77** | teddy bear |
| **38** | tennis racket | **78** | hair drier |
| **39** | bottle | **79** | toothbrush |

---

Key Classes for Restaurant Analytics

| Class ID | Name | Usage in Project |
| :---: | :--- | :--- |
| `0` | **person** | Customer and staff detection, occupancy counting |
| `39` | **bottle** | Table item / beverage tracking |
| `40` | **wine glass** | Dining tracking |
| `41` | **cup** | Beverage service tracking |
| `42 – 45` | **fork, knife, spoon, bowl** | Cutlery & dish tracking |
| `52 – 55` | **hot dog, pizza, donut, cake** | Food delivery / dining

## GitHub project: Smart-Parking-Management-System
This is one of my GitHub repositories: Smart-Parking-Management-System.
Main language: Python.
Code: https://github.com/sidicksino/Smart-Parking-Management-System (updated 2026-10-19, 0 stars).
README:
Smart-Parking-Management-System

## GitHub project: SidickPortfolio
This is one of my GitHub repositories: SidickPortfolio.
Main language: CSS.
Live demo: https://sidick.vercel.app
Code: https://github.com/sidicksino/SidickPortfolio (updated 2026-09-09, 0 stars).
README:
Sidick Sino — Portfolio

Personal portfolio of **Sidick Abdoulaye Hissein** — Data Scientist and
Full-Stack Developer based between N'Djamena, Chad and Kigali, Rwanda.

**Live:** 

A bilingual (English / French) single-page site with a dark and light theme,
featuring shipped projects across web, mobile, design and machine learning.

---

Stack

| | |
|---|---|
| **Framework** | React 19 + Vite 8 (Rolldown) |
| **Routing** | React Router 7 |
| **i18n** | i18next / react-i18next — English + French |
| **Styling** | Hand-written CSS with a design-token system, Tailwind v4 available |
| **Motion** | Framer Motion, GSAP (ScrollTrigger) |
| **Forms** | EmailJS + SweetAlert2 |
| **SEO** | React 19 native document metadata, generated sitemap, robots.txt, hreflang (en/fr) |
| **Hosting** | Vercel |

Running it

Requires Node 18+.

```bash
npm install     # postinstall scripts must be approved — see note below
npm run dev     # http://localhost:5173
```

| Script | Does |
|---|---|
| `npm run dev` | Dev server with HMR |
| `npm run build` | Production build to `dist/` (regenerates `sitemap.xml` first) |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | ESLint |

> **On `npm install`:** this project pins `legacy-peer-deps=true` in `.npmrc`,
> and `package.json` carries an `allowScripts` block for `esbuild`,
> `@tailwindcss/oxide` and `fsevents`. Those postinstall scripts unpack native
> binaries — **the build fails without them**, so keep that block.

Layout

```
src/
├─ index.css              ← design tokens. The single source of truth.
├─ i18n.js                ← i18next setup; also keeps  in sync
├─ App.jsx                ← routes
├─ data/
│  ├─ projectData.js      ← every project + the homepage `featuredProjects` list
│  └─ siteData.js         ← skills, service and category metadata
├─ locales/               ← en.json / fr.json (keys must stay in parity)
└─ components/
   ├─ hero/  about/  skills/       ← homepage sections
   ├─ featured/                    ← the featured-work grid
   ├─ project/  projects/          ← category cards + per-category pages
   └─ contact/  footer/  navbar/
```

Three conventions worth knowing

**All design tokens live in `src/index.css`.** No other stylesheet may declare
`:root`. Colours, spacing, type and radii are tokens — component CSS consumes
them and never hard-codes a hex value.

**Content is data, not markup.** Projects, skills and services are defined in
`src/data/`, and every user-facing string i

## GitHub project: SnapChef
This is one of my GitHub repositories: SnapChef.
Main language: TypeScript.
Code: https://github.com/sidicksino/SnapChef (updated 2026-09-07, 0 stars).
README:
SnapChef 🧑‍🍳📸

SnapChef is an intelligent mobile application that allows users to snap a photo of the ingredients in their fridge and instantly generates a delicious recipe using an AI vision model (YOLOv8 Edge) and a Large Language Model (OpenAI GPT-4o).

Project Structure

```text
fridge-recipe-generator/
│
├── backend/                  # FastAPI Backend API
│   ├── main.py               # Application entry point
│   ├── database.py           # SQLAlchemy setup and connection
│   ├── models.py             # Database ORM models (User, Recipe, etc.)
│   ├── schemas.py            # Pydantic validation schemas
│   ├── security.py           # Password hashing & JWT token logic
│   ├── deps.py               # Dependency injection (e.g., get_current_user)
│   ├── requirements.txt      # Python dependencies
│   └── routers/              # API Endpoints
│       ├── auth.py           # Login, Register, Password Management
│       ├── users.py          # Profile management
│       └── recipes.py        # LLM Generation & History
│
├── ml/                       # Machine Learning / Computer Vision
│   ├── train_colab.ipynb     # Google Colab notebook for training YOLOv8n
│   └── train.py              # Local training script (alternative to Colab)
│
├── mobile/                   # Expo (React Native) mobile app
│   ├── src/app/               # expo-router file-based routes
│   ├── src/components/        # Shared UI components
│   ├── src/global.css         # Tailwind directives (Nativewind)
│   ├── tailwind.config.js     # Nativewind/Tailwind config
│   └── app.json                # Expo app config
│
├── .gitignore                # Git ignore rules
├── AGENT.md                  # Project context and rules for AI Agent
├── implementation_plan.md    # Master architecture and API design document
└── task.md                   # Checklist of completed and pending tasks
```

How to Run the Backend

Follow these steps to run the backend API server on your local machine.

1. Prerequisites
- Python 3.10+ installed on your Mac (`python3`).

2. Setup the Virtual Environment
Open your terminal, navigate to the `backend` folder, and create a virtual environment to isolate the dependencies:

```bash
cd backend
python3 -m venv venv
```

3. Activate the Environment
Activate the virtual environment. **You must do this every time you open a new terminal.**

```bash
source venv/bin/activate
```
*(Your terminal prompt should now have `(venv)` at the beginning).*

4. Install Dependencies

## GitHub project: Final-Project-Project-Trainning
This is one of my GitHub repositories: Final-Project-Project-Trainning.
Code: https://github.com/sidicksino/Final-Project-Project-Trainning (updated 2026-08-26, 0 stars).
README:
Final-Project-Project-Trainning

## GitHub project: E-commerce-power-bi-dashboard
This is one of my GitHub repositories: E-commerce-power-bi-dashboard.
Code: https://github.com/sidicksino/E-commerce-power-bi-dashboard (updated 2026-08-20, 0 stars).
README:
E-commerce-power-bi-dashboard

## GitHub project: Coca-cola-power-bi-dashboard
This is one of my GitHub repositories: Coca-cola-power-bi-dashboard.
Code: https://github.com/sidicksino/Coca-cola-power-bi-dashboard (updated 2026-08-20, 0 stars).
README:
Coca-cola-power-bi-dashboard

## GitHub project: training-Reactjs-Nodejs-MongoDb
This is one of my GitHub repositories: training-Reactjs-Nodejs-MongoDb.
Main language: JavaScript.
Live demo: https://post-trainning.vercel.app
Code: https://github.com/sidicksino/training-Reactjs-Nodejs-MongoDb (updated 2026-08-09, 0 stars).
README:
training-Reactjs-Nodejs-MongoDb

## GitHub project: 30-Day-Excel-Data-Analysis-Challenge
This is one of my GitHub repositories: 30-Day-Excel-Data-Analysis-Challenge.
Code: https://github.com/sidicksino/30-Day-Excel-Data-Analysis-Challenge (updated 2026-08-09, 0 stars).
README:
30-Day Excel Data Analysis Challenge

A 30-day journey to improve my skills in:

Data Cleaning
Data Analysis
Excel
PivotTables
Data Visualization
Dashboard Design
Business Intelligence

Projects

| Day | Project | Industry | Status |
|---|---|---|---|
| 01 | NovaMart Retail | Retail | ✅ |
| 02 | ... | ... | ⬜ |
| 03 | ... | ... | ⬜ |
| ... | ... | ... | ... |
| 30 | ... | ... | ⬜ |

## GitHub project: training
This is one of my GitHub repositories: training.
Main language: CSS.
Live demo: https://training-wine-six.vercel.app
Code: https://github.com/sidicksino/training (updated 2026-08-02, 0 stars).
README:
training

## GitHub project: AI
This is one of my GitHub repositories: AI.
Main language: JavaScript.
Live demo: https://sinoaiserver.vercel.app
Code: https://github.com/sidicksino/AI (updated 2026-08-02, 0 stars).

## GitHub project: abdraman
This is one of my GitHub repositories: abdraman.
Main language: TypeScript.
Live demo: https://abdraman.vercel.app
Code: https://github.com/sidicksino/abdraman (updated 2026-07-13, 0 stars).
README:
Abdraman Ibrahim Nassour — Portfolio

A personal portfolio website showcasing who I am, my skills, my work, my academic
journey, and how to get in touch.

Sections
- **Hero** — intro, name, and quick links
- **About** — Who I Am / What I Do / My Mission
- **Skills** — Frontend, Backend, Database, Mobile, Tools & DevOps
- **My Work** — selected projects
- **My Journey** — education timeline (2017 → Master's in IT)
- **Contact** — contact form + direct email

Tech Stack

| Purpose | Tool |
|---------|------|
| Framework | Next.js 16 (Turbopack) |
| UI library | React 19 |
| Language | TypeScript |
| Styling | Tailwind CSS v4 |
| Animation | Framer Motion |
| Icons | Lucide React |
| Class helpers | clsx, tailwind-merge |
| Linting | ESLint |
| Runtime / packages | Node.js + npm |

**In one sentence:** a portfolio built with **Next.js + React + TypeScript**,
styled with **Tailwind CSS**, animated with **Framer Motion**, using **Lucide** icons.

Getting Started

Install dependencies (run once):

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open http://localhost:3000 in your browser.

Commands

| Command | What it does |
|---------|--------------|
| `npm install` | Install the tools/packages |
| `npm run dev` | Start the site while working |
| `npm run build` | Build the final version for the internet |
| `npm run start` | Run the built version |
| `npm run lint` | Check code for mistakes |

Project Structure

See STRUCTURE.md for a full explanation of the folders and files.

## GitHub project: predictacart-app-group-work
This is one of my GitHub repositories: predictacart-app-group-work.
Main language: HTML.
Code: https://github.com/sidicksino/predictacart-app-group-work (updated 2026-06-26, 0 stars).
README:
predictacart-app-group-work

## GitHub project: Spam-Detection-Azure-v1
This is one of my GitHub repositories: Spam-Detection-Azure-v1.
Main language: HTML.
Code: https://github.com/sidicksino/Spam-Detection-Azure-v1 (updated 2026-06-23, 0 stars).
README:
Spam-Detection-Azure-v1

> **AI-powered spam detection** — A FastAPI web app that classifies messages as **Spam** or **Ham** using a Multinomial Naive Bayes model deployed on Azure ML.

🚀 Live Demo

**https://spam-detection-svis.onrender.com/**

---

🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | Multinomial Naive Bayes (scikit-learn) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML / CSS / JavaScript (glassmorphism UI) |
| **ML Endpoint** | Azure Machine Learning Online Endpoint |
| **Deployment** | Render (Docker) |
| **Container** | Dockerfile |

✨ Features

- 🔍 Real-time spam detection via Azure ML inference
- 🎨 Premium dark-mode glassmorphism UI with animated background
- ⚡ Fast API response with loading states
- 📱 Fully responsive design
- 🐳 Dockerized for easy deployment

📂 Project Structure

```
spam-app/
├── main.py                # FastAPI app + /predict endpoint
├── templates/
│   └── index.html         # Frontend UI
├── ML-Cloud-Azure/        # Azure ML deployment docs & guides
├── spam.csv               # Training dataset
├── Dockerfile             # Container config
├── requirements.txt       # Python dependencies
└── .env                   # Azure ML endpoint URL & key (not committed)
```

🏃 Local Setup

```bash
Clone the repo
git clone https://github.com/sidicksino/Spam-Detection-Azure-v1.git
cd Spam-Detection-Azure-v1

Create virtual environment
python -m venv venv
source venv/bin/activate

Install dependencies
pip install -r requirements.txt

Set environment variables (.env)
AZURE_ENDPOINT_URL=
AZURE_API_KEY=

Run
uvicorn main:app --reload
```

📜 License

MIT

## GitHub project: Spam-Detection
This is one of my GitHub repositories: Spam-Detection.
Main language: HTML.
Code: https://github.com/sidicksino/Spam-Detection (updated 2026-06-16, 0 stars).
README:
Spam Detection Web Application

A beginner-friendly **MLOps demo project**. This project demonstrates the full lifecycle of a machine learning application — from training a model to deploying it on the cloud.

>  Follow this README step by step. By the end, you will have your own spam detector running on the internet!

---

What You Will Build

A web app that classifies text messages as **Spam** or **Not Spam** using machine learning.

| Component | Technology |
|---|---|
| Machine Learning | Multinomial Naive Bayes |
| Feature Extraction | CountVectorizer (scikit-learn) |
| Web Framework | Flask (Python) |
| Frontend | HTML5 + CSS3 |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Option A | AWS EC2 |
| Cloud Option B | Azure Web Apps |

---

Project Structure

```
Spam-Detection/
│
├── app.py                   # Flask web application (runs the website)
├── train.py                 # Trains the ML model and saves it
├── model.pkl                # Trained model file (created after running train.py)
├── vectorizer.pkl           # Text vectorizer file (created after running train.py)
├── requirements.txt         # Python packages needed
├── Dockerfile               # Instructions to build a Docker container
├── .gitignore               # Files Git should ignore
├── README.md                # This file
│
├── templates/
│   └── index.html           # The web page (HTML + CSS)
│
└── .github/
    └── workflows/
        └── docker-build.yml # Automatic Docker build & push (GitHub Actions)
```

---

PART 1 — Clone & Run Locally

These steps get the project running on your own computer.

---

Prerequisites

Make sure you have these installed before starting:

| Tool | How to check | Download |
|---|---|---|
| Python 3.9+ | `python3 --version` | https://www.python.org |
| pip | `pip --version` | Comes with Python |
| Git | `git --version` | https://git-scm.com |

---

Step 1 — Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/sidicksino/Spam-Detection.git
cd Spam-Detection
```

You should now see the project files in your terminal.

---

Step 2 — The Dataset is Already Included

The `spam.csv` file is included in the repository, so you get it automatically when you clone. Nothing extra to do!

> The dataset contains **5,572 SMS messages** labelled as `ham` (legitimate) or `spam`.

---

Step 3 — Create a Virtual Environment

A virtual environment keeps your project's packages separate from the rest of your comput

## GitHub project: SmartGuard
This is one of my GitHub repositories: SmartGuard.
Main language: JavaScript.
Code: https://github.com/sidicksino/SmartGuard (updated 2026-06-04, 0 stars).

## GitHub project: duolingo-clone
This is one of my GitHub repositories: duolingo-clone.
Main language: TypeScript.
Code: https://github.com/sidicksino/duolingo-clone (updated 2026-05-18, 0 stars).

## GitHub project: Snapchat_bot
This is one of my GitHub repositories: Snapchat_bot.
Main language: JavaScript.
Code: https://github.com/sidicksino/Snapchat_bot (updated 2026-04-30, 0 stars).
README:
💎 Luxury Snap Engine (Snapchat Bot)

A high-end cinematic image generation bot that uses AI to create hyper-realistic "luxury lifestyle" shots and sends them to Telegram. It combines the power of **Google Gemini** for intelligent prompt/caption engineering and **Hugging Face (Flux.1-Schnell)** for state-of-the-art image generation.

🚀 Features

- **Luxury Content Engine**: Randomly selects cinematic scenes (penthouses, supercars, luxury lobbies, etc.).
- **Anti-AI Prompting**: Uses Gemini to engineer prompts that avoid the "AI look," focusing on real-world photography techniques (35mm lens, natural grain, depth of field).
- **Human Captions**: Generates short, lowercase, Snapchat-style captions that feel authentic.
- **Automated Workflow**: 
    1. Generates a master prompt.
    2. Generates a high-resolution image.
    3. Crafts a natural caption.
    4. Delivers everything to Telegram.

🛠️ Tech Stack

- **Logic**: Node.js
- **LLM**: Google Gemini Flash (Prompts & Captions)
- **Image Gen**: Flux.1-Schnell via Hugging Face Inference API
- **Communication**: Telegram Bot API

📋 Prerequisites

- Node.js installed
- A Telegram Bot token (from @BotFather)
- A Google AI Studio API Key
- A Hugging Face Access Token

⚙️ Setup

1. **Clone the repository**:
   ```bash
   git clone 
   cd SnapChat
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_key
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   HF_TOKEN=your_hugging_face_token
   ```

🏃 Usage

To run the engine once:
```bash
npm start
```
or
```bash
npm run main
```

📜 License
ISC

## GitHub project: MongoDB-course
This is one of my GitHub repositories: MongoDB-course.
Main language: Python.
Live demo: https://mongo-db-course.vercel.app
Code: https://github.com/sidicksino/MongoDB-course (updated 2026-03-13, 0 stars).
README:
MongoDB-course

## GitHub project: SinoAgency
This is one of my GitHub repositories: SinoAgency.
Main language: HTML.
Code: https://github.com/sidicksino/SinoAgency (updated 2026-02-28, 0 stars).
README:
SinoAgency

## GitHub project: Boutique_Frontend
This is one of my GitHub repositories: Boutique_Frontend.
Main language: JavaScript.
Code: https://github.com/sidicksino/Boutique_Frontend (updated 2026-02-26, 0 stars).

## GitHub project: wallet
This is one of my GitHub repositories: wallet.
Main language: JavaScript.
Code: https://github.com/sidicksino/wallet (updated 2026-02-26, 0 stars).
README:
wallet

## GitHub project: AAPT
This is one of my GitHub repositories: AAPT.
Main language: TypeScript.
Live demo: https://aapt-eight.vercel.app
Code: https://github.com/sidicksino/AAPT (updated 2026-02-26, 0 stars).
README:
AAPT - Association Actions Pour Tous

An interactive, responsive web application for the AAPT non-profit organization, built with modern web technologies.

🛠 Tech Stack
- **Framework**: React 18 + Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4 + Vanilla CSS variables
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Backend/Auth**: Supabase
- **Internationalization**: i18next

📂 Project Structure
After the 2026 Refactor, the project follows a standard `src/` directory structure:

```
src/
├── components/     # Reusable UI components (Navbar, Footer, etc.)
├── context/        # React Context (AuthContext)
├── lib/            # Utilities & Clients (Supabase client)
├── pages/          # Page views (Home, About, News, Admin...)
├── App.tsx         # Main Application Component & Routing
├── main.tsx        # Entry point
└── index.css       # Global styles & Tailwind directives
```

🚀 Getting Started

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Environment Setup**
   Ensure you have a `.env.local` file with:
   ```env
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_ANON_KEY=your_supabase_key
   ```

3. **Run Development Server**
   ```bash
   npm run dev
   ```

4. **Build for Production**
   ```bash
   npm run build
   ```

🔐 Admin Access
The admin section is protected via Supabase Auth.
- Login Route: `/admin/login`
- Protected Routes: Wrapped in `` component.

🌍 Internationalization
Translations are stored in `public/locales/{lang}/translation.json`.
The app uses `i18next-browser-languagedetector` to automatically detect user language.

## GitHub project: covid-19-dashboard
This is one of my GitHub repositories: covid-19-dashboard.
Main language: JavaScript.
Code: https://github.com/sidicksino/covid-19-dashboard (updated 2026-02-23, 1 stars).

## GitHub project: Alarm
This is one of my GitHub repositories: Alarm.
Main language: TypeScript.
Code: https://github.com/sidicksino/Alarm (updated 2026-02-20, 0 stars).

## GitHub project: ZAHO
This is one of my GitHub repositories: ZAHO.
Main language: CSS.
Live demo: https://zaho.vercel.app
Code: https://github.com/sidicksino/ZAHO (updated 2026-02-19, 0 stars).
README:
ZAHO

## GitHub project: abakar_ibrahim_haroun
This is one of my GitHub repositories: abakar_ibrahim_haroun.
Main language: CSS.
Code: https://github.com/sidicksino/abakar_ibrahim_haroun (updated 2026-02-16, 0 stars).
README:
abakar_ibrahim_haroun

## GitHub project: python6h
This is one of my GitHub repositories: python6h.
Main language: Python.
Code: https://github.com/sidicksino/python6h (updated 2026-02-16, 0 stars).

## GitHub project: Ahmat-Fat
This is one of my GitHub repositories: Ahmat-Fat.
Main language: HTML.
Code: https://github.com/sidicksino/Ahmat-Fat (updated 2026-02-11, 0 stars).
README:
Ahmat-Fat

## GitHub project: Data-Visualization-Fat
This is one of my GitHub repositories: Data-Visualization-Fat.
Main language: JavaScript.
Live demo: https://data-visualization-fat.vercel.app
Code: https://github.com/sidicksino/Data-Visualization-Fat (updated 2026-02-11, 0 stars).

## GitHub project: tchadinfos_V2
This is one of my GitHub repositories: tchadinfos_V2.
Main language: TypeScript.
Code: https://github.com/sidicksino/tchadinfos_V2 (updated 2026-02-08, 0 stars).

## GitHub project: CinematicConstructionBot
This is one of my GitHub repositories: CinematicConstructionBot.
Main language: JavaScript.
Code: https://github.com/sidicksino/CinematicConstructionBot (updated 2026-02-06, 0 stars).
README:
Cinematic Construction Bot 🏗️🎬

A Telegram bot that generates a cinematic architectural construction workflow using AI.

Features
- **Interactive Selection**: Choose from 10 architectural styles.
- **AI Prompt Generation**: Uses Google Gemini to create detailed, consistent prompts.
- **Photorealistic Images**: Generates 6 sequential construction images using FLUX.1 (via Hugging Face).
- **Video Prompts**: Provides detained image-to-video prompts for animation.

Setup

1.  **Prerequisites**:
    - Node.js installed.
    - API Keys in `.env`: `GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `HF_TOKEN`.

2.  **Configuration**:
    The project is located in `SnapChat/CinematicConstructionBot`.
    Dependencies are already installed.

How to Run

1.  Open a terminal.
2.  Navigate to the project folder:
    ```bash
    cd /Users/abakar/Desktop/project/SnapChat/CinematicConstructionBot
    ```
3.  Start the bot:
    ```bash
    node index.js
    ```
4.  **Usage**:
    - Open your Telegram bot.
    - Type `/start`.
    - Select a structure number (e.g., `1`).
    - Wait for the magic! 🎥

Tech Stack
- **Framework**: Telegraf (Node.js)
- **LLM**: Google Gemini 1.5 Flash
- **Image Gen**: FLUX.1-dev (Hugging Face Inference API)
CinematicConstructionBot

## GitHub project: tchadInfos_version1
This is one of my GitHub repositories: tchadInfos_version1.
Main language: TypeScript.
Code: https://github.com/sidicksino/tchadInfos_version1 (updated 2026-01-30, 0 stars).

## GitHub project: School
This is one of my GitHub repositories: School.
Main language: TypeScript.
Live demo: https://school-rosy-nine.vercel.app
Code: https://github.com/sidicksino/School (updated 2026-01-29, 0 stars).
README:
Académie Royale

A modern, futuristic website for a fictional academy, built with React, Vite, and TailwindCSS (via standard CSS or utility classes if present).

  

✨ Features

- **Modern UI/UX**: Animations with Framer Motion, responsive design.
- **Internationalization**: Multi-language support (Français/English).
- **Dark/Light Mode**: Theme switching capability.
- **Routing**: Client-side routing with React Router.

🛠️ Tech Stack

- **Framework**: React 18+ with Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS (presumed based on class names) / CSS
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Testing**: Vitest + React Testing Library
- **Linting**: ESLint + Prettier

🚀 Getting Started

Prerequisites

- Node.js (v18 or higher recommended)
- npm or yarn

Installation

1.  **Clone the repository:**
    ```bash
    git clone 
    cd académie-royale
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Environment Setup:**
    Duplicate `.env.example` to `.env.local` and configure your keys.
    ```bash
    cp .env.example .env.local
    ```
    - `VITE_GEMINI_API_KEY`: API key for Gemini features (if any).

Running the App

Start the development server:
```bash
npm run dev
```
Open http://localhost:5173 to view it in the browser.

📜 Scripts

- `npm run dev`: Start dev server
- `npm run build`: Build for production
- `npm run preview`: Preview production build
- `npm run lint`: Run ESLint check
- `npm run format`: Format code with Prettier
- `npm test`: Run tests with Vitest

📂 Project Structure

```
src/
├── components/     # Reusable UI components and sections
├── contexts/       # React Context providers (Theme, Language)
├── lib/            # Utilities and translations
├── ui/             # Generic UI elements
├── App.tsx         # Main application component
└── main.tsx        # Entry point
```

🤝 Contributing

1.  Fork the repo
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## GitHub project: First-Django
This is one of my GitHub repositories: First-Django.
Main language: CSS.
Code: https://github.com/sidicksino/First-Django (updated 2026-01-27, 0 stars).

## GitHub project: animation
This is one of my GitHub repositories: animation.
Main language: TypeScript.
Code: https://github.com/sidicksino/animation (updated 2026-01-26, 0 stars).

## GitHub project: SinoPharmacie
This is one of my GitHub repositories: SinoPharmacie.
Main language: HTML.
Code: https://github.com/sidicksino/SinoPharmacie (updated 2026-01-20, 0 stars).
README:
SinoPharmacie

## GitHub project: djangoDev
This is one of my GitHub repositories: djangoDev.
Main language: Python.
Code: https://github.com/sidicksino/djangoDev (updated 2026-01-02, 0 stars).
README:
djangoDev

## GitHub project: test
This is one of my GitHub repositories: test.
Main language: TypeScript.
Live demo: https://test-azure-three-52.vercel.app
Code: https://github.com/sidicksino/test (updated 2025-12-28, 0 stars).

## GitHub project: ML-project
This is one of my GitHub repositories: ML-project.
Main language: Jupyter Notebook.
Code: https://github.com/sidicksino/ML-project (updated 2025-11-30, 0 stars).
README:
ML-project

## GitHub project: Disease-Risk-Prediction
This is one of my GitHub repositories: Disease-Risk-Prediction.
Main language: Jupyter Notebook.
Code: https://github.com/sidicksino/Disease-Risk-Prediction (updated 2025-11-16, 0 stars).
README:
Disease-Risk-Prediction

## GitHub project: machine_learning_lecture
This is one of my GitHub repositories: machine_learning_lecture.
Main language: Jupyter Notebook.
Code: https://github.com/sidicksino/machine_learning_lecture (updated 2025-11-14, 0 stars).
README:
---
title: "Pima Diabetes ML Project Setup and Deployment"
author: "Sidick Abdoulaye Sino"
date: "2025-11-13"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

1. Project Overview

This document explains the full workflow to set up, train, and deploy the **Pima Indians Diabetes ML model** using **Streamlit** and **GitHub**.

---

2. GitHub Setup

1. Initialize local Git repository:

```bash
git init
```

2. Add README.md and commit:

```bash
echo "# machine_learning_lecture" >> README.md
git add README.md
git commit -m "first commit"
```

3. Add remote repository and push:

```bash
git branch -M main
git remote add origin https://github.com/sidicksino/machine_learning_lecture.git
git push -u origin main
```

4. Add project files and commit:

```bash
git add .
git commit -m "Add Streamlit app, dataset, and notebook"
git push origin main
```

---

3. Python Environment Setup

1. Create virtual environment:

```bash
python3 -m venv venv
```

2. Activate virtual environment:

**Mac/Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install streamlit scikit-learn pandas numpy joblib watchdog
```

4. Optional: save dependencies to `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

4. Training and Saving Model

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

Load data
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Train model
model = LogisticRegression()
model.fit(X_scaled, y)

Save model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

5. Streamlit App

Create `app.py`:

```python
import streamlit as st
import joblib
import pandas as pd
import numpy as np

Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Pima Diabetes Prediction")

Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose")
blood_pressure = st.number_input("BloodPressure")
skin_thickness = st.number_input("SkinThickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age", min_value=0)

Prepa

## GitHub project: Mouvies_recommendation
This is one of my GitHub repositories: Mouvies_recommendation.
Main language: Jupyter Notebook.
Code: https://github.com/sidicksino/Mouvies_recommendation (updated 2025-10-08, 1 stars).

## GitHub project: SinoCoffee
This is one of my GitHub repositories: SinoCoffee.
Main language: TypeScript.
Live demo: https://sino-coffee.vercel.app
Code: https://github.com/sidicksino/SinoCoffee (updated 2025-10-06, 0 stars).

## GitHub project: SignUpSignIn-Django
This is one of my GitHub repositories: SignUpSignIn-Django.
Main language: Python.
Code: https://github.com/sidicksino/SignUpSignIn-Django (updated 2025-09-26, 0 stars).

## GitHub project: insta_abeche
This is one of my GitHub repositories: insta_abeche.
Main language: HTML.
Code: https://github.com/sidicksino/insta_abeche (updated 2025-09-21, 0 stars).

## GitHub project: sino_Ai
This is one of my GitHub repositories: sino_Ai.
Main language: JavaScript.
Code: https://github.com/sidicksino/sino_Ai (updated 2025-08-25, 0 stars).

## GitHub project: Boutique_Backend
This is one of my GitHub repositories: Boutique_Backend.
Main language: JavaScript.
Code: https://github.com/sidicksino/Boutique_Backend (updated 2025-08-24, 0 stars).

## GitHub project: test-agura
This is one of my GitHub repositories: test-agura.
Main language: JavaScript.
Code: https://github.com/sidicksino/test-agura (updated 2025-08-13, 0 stars).

## GitHub project: Agura_Tickets_Backend
This is one of my GitHub repositories: Agura_Tickets_Backend.
Code: https://github.com/sidicksino/Agura_Tickets_Backend (updated 2025-07-26, 0 stars).

## GitHub project: Boutique_sino
This is one of my GitHub repositories: Boutique_sino.
Main language: TypeScript.
Code: https://github.com/sidicksino/Boutique_sino (updated 2025-07-21, 0 stars).

## GitHub project: team_work
This is one of my GitHub repositories: team_work.
Main language: HTML.
Code: https://github.com/sidicksino/team_work (updated 2025-06-18, 0 stars).

## GitHub project: sidick_abdoulaye_sino1
This is one of my GitHub repositories: sidick_abdoulaye_sino1.
Main language: CSS.
Code: https://github.com/sidicksino/sidick_abdoulaye_sino1 (updated 2025-04-27, 0 stars).

## GitHub project: sidick_abdoulaye_sino
This is one of my GitHub repositories: sidick_abdoulaye_sino.
Main language: HTML.
Code: https://github.com/sidicksino/sidick_abdoulaye_sino (updated 2025-01-08, 0 stars).

## GitHub project: sidick_sino
This is one of my GitHub repositories: sidick_sino.
Main language: HTML.
Code: https://github.com/sidicksino/sidick_sino (updated 2025-01-02, 0 stars).
