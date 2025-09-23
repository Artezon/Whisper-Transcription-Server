import Fastify from "fastify";
import fastifyStatic from "@fastify/static";
import fastifyView from "@fastify/view";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import ejs from "ejs";

dotenv.config();
const fastify = Fastify({ logger: true });
global.projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

await fastify.register(fastifyStatic, {
  root: path.join(global.projectRoot, "public"),
  prefix: "/transcribe/public/",
});

await fastify.register(fastifyView, {
  engine: { ejs },
  root: path.join(global.projectRoot, "views"),
});

async function checkBackendAvailability() {
  try {
    const res = await fetch(`${process.env.BACKEND_ORIGIN}/api/status`, {
      signal: AbortSignal.timeout(1000),
    });
    const data = await res.json();
    return data.ready === true;
  } catch (err) {
    return false;
  }
}

fastify.get("/transcribe", async (request, reply) => {
  if (await checkBackendAvailability()) {
    return reply.view("index.ejs", { backendOrigin: process.env.BACKEND_ORIGIN });
  } else {
    return reply.view("unavailable.ejs", {
      backendOrigin: process.env.BACKEND_ORIGIN,
      unavailableMessage: process.env.BACKEND_UNAVAILABLE_HTML,
    });
  }
});

fastify.get("/transcribe/transcription/:task_id", async (request, reply) => {
  if (await checkBackendAvailability()) {
    return reply.view("transcription.ejs", {
      backendOrigin: process.env.BACKEND_ORIGIN,
    });
  } else {
    return reply.view("unavailable.ejs", {
      backendOrigin: process.env.BACKEND_ORIGIN,
      unavailableMessage: process.env.BACKEND_UNAVAILABLE_HTML,
    });
  }
});

fastify.get("/transcribe/404", async (request, reply) => {
  return reply.view("404.ejs");
});

try {
  await fastify.listen({
    host: process.env.HOST || "localhost",
    port: parseInt(process.env.PORT || "8080"),
  });
  console.log(`Server running`);
} catch (error) {
  console.error("Error starting server:", error);
  process.exit(1);
}
