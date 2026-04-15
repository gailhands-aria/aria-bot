const express = require("express");
const app = express();

let queue = [];

app.get("/", (req, res) => {
  const user = req.query.user || "unknown";
  const msg = req.query.msg || "";

  if (!msg.toLowerCase().startsWith("!aria")) {
    return res.send("ignored");
  }

  const cleaned = msg.replace("!aria", "").trim();

  const reply = `${user}, I hear you... ${cleaned}`;

  queue.push({
    user,
    reply,
    time: Date.now()
  });

  res.send("queued");
});

app.get("/next", (req, res) => {
  if (queue.length === 0) {
    return res.send("");
  }

  const item = queue.shift();
  res.send(item.reply);
});

app.listen(3000, () => {
  console.log("Aria running");
});
