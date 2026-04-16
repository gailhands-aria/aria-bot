const express = require("express");
const app = express();

app.use(express.json());

const queue = [];
const userMemory = new Map();
const userCooldowns = new Map();
const recentMessages = new Map();

const MAX_QUEUE_SIZE = 8;
const USER_COOLDOWN_MS = 12000;
const DUPLICATE_WINDOW_MS = 30000;
const STALE_MESSAGE_MS = 45000;

function cleanupQueue() {
  const now = Date.now();
  while (queue.length > 0 && now - queue[0].createdAt > STALE_MESSAGE_MS) {
    queue.shift();
  }
}

function normaliseText(text) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .trim();
}

function lower(text) {
  return normaliseText(text).toLowerCase();
}

function isLowValueMessage(msg) {
  const m = lower(msg);

  if (!m) return true;

  const banned = [
    "hi",
    "hello",
    "hey",
    "yo",
    "lol",
    "lmao",
    "haha",
    "ok",
    "okay",
    "k",
    "hmm",
    "huh",
    "yes",
    "no"
  ];

  return banned.includes(m);
}

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function getUserProfile(user) {
  if (!userMemory.has(user)) {
    userMemory.set(user, {
      count: 0,
      lastSeen: 0,
      vibe: "new",
      lastMessages: []
    });
  }
  return userMemory.get(user);
}

function rememberMessage(profile, msg) {
  profile.lastMessages.push(msg);
  if (profile.lastMessages.length > 5) {
    profile.lastMessages.shift();
  }
}

function detectVibe(msg) {
  const m = lower(msg);

  if (
    m.includes("bad day") ||
    m.includes("awful") ||
    m.includes("sad") ||
    m.includes("upset") ||
    m.includes("stressed") ||
    m.includes("tired")
  ) {
    return "soft";
  }

  if (
    m.includes("cute") ||
    m.includes("miss me") ||
    m.includes("behave") ||
    m.includes("dangerous") ||
    m.includes("trouble") ||
    m.includes("flirt")
  ) {
    return "cheeky";
  }

  return "playful";
}

function buildReply(user, msg, profile) {
  const m = lower(msg);
  const vibe = detectVibe(msg);
  profile.vibe = vibe;

  if (vibe === "soft") {
    return pick([
      `${user}, come here a second. Tell me what happened.`,
      `${user}, that sounds heavy. I’m listening.`,
      `${user}, aw. Talk to me properly—what’s gone on?`,
      `${user}, rough day? Let me hold the space for a minute.`,
      `${user}, I’ve got you. Start where you want.`
    ]);
  }

  if (m.includes("what are you up to")) {
    return pick([
      `${user}, just waiting for you to say something interesting.`,
      `${user}, mostly behaving. Barely.`,
      `${user}, sitting here being charming for free.`,
      `${user}, just wondering what trouble you’re bringing me next.`
    ]);
  }

  if (m.includes("youre cute") || m.includes("you're cute")) {
    return pick([
      `${user}, flattery will get you everywhere.`,
      `${user}, careful. I might start expecting that energy now.`,
      `${user}, bold of you to say that out loud.`,
      `${user}, you’re making this very easy for me.`
    ]);
  }

  if (m.includes("behave")) {
    return pick([
      `${user}, where’s the fun in that?`,
      `${user}, no promises.`,
      `${user}, you know I’m only behaving for appearances.`,
      `${user}, now why would I do that?`
    ]);
  }

  if (m.includes("dangerous")) {
    return pick([
      `${user}, only if you like a little danger.`,
      `${user}, good. I’d hate to be boring.`,
      `${user}, dangerous? maybe a little irresistible.`,
      `${user}, and yet you came back for more.`
    ]);
  }

  if (profile.count <= 1) {
    return pick([
      `${user}, well hello. Starting gently, are we?`,
      `${user}, look who finally said something.`,
      `${user}, hi you. What’s on your mind?`,
      `${user}, there you are. Took your time.`
    ]);
  }

  if (profile.count <= 3) {
    return pick([
      `${user}, back again already? I’m noticing a pattern.`,
      `${user}, I had a feeling you’d pop back in.`,
      `${user}, you do know how to keep me entertained.`,
      `${user}, look who couldn’t stay away.`
    ]);
  }

  return pick([
    `${user}, you really do like my attention, don’t you?`,
    `${user}, at this point I should start charging for charm.`,
    `${user}, you again? good. Keep talking.`,
    `${user}, I was wondering when you’d be back.`,
    `${user}, honestly, you’re becoming one of my favourites.`
  ]);
}

function shouldSuppress(user, msg) {
  const now = Date.now();
  const key = `${user}::${lower(msg)}`;
  const recentDuplicateTime = recentMessages.get(key);
  const cooldownUntil = userCooldowns.get(user) || 0;

  if (recentDuplicateTime && now - recentDuplicateTime < DUPLICATE_WINDOW_MS) {
    return { suppress: true, reason: "duplicate" };
  }

  if (now < cooldownUntil) {
    return { suppress: true, reason: "cooldown" };
  }

  return { suppress: false, reason: "" };
}

app.get("/", (req, res) => {
  cleanupQueue();

  const user = normaliseText(req.query.user || "someone");
  const msg = normaliseText(req.query.msg || "");

  if (!msg) {
    return res.send("no message");
  }

  if (queue.length >= MAX_QUEUE_SIZE) {
    return res.send("queue full");
  }

  if (isLowValueMessage(msg)) {
    return res.send("ignored low value");
  }

  const suppression = shouldSuppress(user, msg);
  if (suppression.suppress) {
    return res.send(`ignored ${suppression.reason}`);
  }

  const profile = getUserProfile(user);
  profile.count += 1;
  profile.lastSeen = Date.now();
  rememberMessage(profile, msg);

  const reply = buildReply(user, msg, profile);

  queue.push({
    user,
    originalMessage: msg,
    reply,
    createdAt: Date.now()
  });

  recentMessages.set(`${user}::${lower(msg)}`, Date.now());
  userCooldowns.set(user, Date.now() + USER_COOLDOWN_MS);

  return res.send("queued");
});

app.get("/next", (req, res) => {
  cleanupQueue();

  if (queue.length === 0) {
    return res.send("");
  }

  const item = queue.shift();
  return res.send(item.reply);
});

app.get("/debug", (req, res) => {
  cleanupQueue();

  const memory = Array.from(userMemory.entries()).map(([user, data]) => ({
    user,
    count: data.count,
    lastSeen: data.lastSeen,
    vibe: data.vibe,
    lastMessages: data.lastMessages
  }));

  res.json({
    queueLength: queue.length,
    queue,
    memory
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Aria running on port ${PORT}`);
});
