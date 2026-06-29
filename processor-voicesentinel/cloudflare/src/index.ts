import { Container, getContainer } from "@cloudflare/containers";

const PROCESSOR_PORT = 28472;

export class ProcessorContainer extends Container {
  defaultPort = PROCESSOR_PORT;
  sleepAfter = "24h";
  enableInternet = true;
  pingEndpoint = "/health";
}

interface Env {
  PROCESSOR: DurableObjectNamespace<ProcessorContainer>;
  STORAGE?: R2Bucket;
  SERVER_KEY?: string;
  POOL_SERVER?: string;
  R2_BUCKET_NAME?: string;
  R2_ACCOUNT_ID?: string;
  R2_ACCESS_KEY_ID?: string;
  R2_SECRET_ACCESS_KEY?: string;
  WHISPER_MODEL?: string;
  TRANSCRIPTION_MODEL?: string;
  REPORT_BUFFER_ENABLED?: string;
  REPORT_BUFFER_SAVE_AUDIO?: string;
}

function containerEnv(env: Env): Record<string, string> {
  const vars: Record<string, string> = {};

  if (env.SERVER_KEY) {
    vars.SERVER_KEY = env.SERVER_KEY;
  }
  if (env.POOL_SERVER) {
    vars.POOL_SERVER = env.POOL_SERVER;
  }
  if (env.TRANSCRIPTION_MODEL) {
    vars.TRANSCRIPTION_MODEL = env.TRANSCRIPTION_MODEL;
  } else if (env.WHISPER_MODEL) {
    vars.TRANSCRIPTION_MODEL = env.WHISPER_MODEL;
  }
  if (env.REPORT_BUFFER_ENABLED) {
    vars.REPORT_BUFFER_ENABLED = env.REPORT_BUFFER_ENABLED;
  }
  if (env.REPORT_BUFFER_SAVE_AUDIO) {
    vars.REPORT_BUFFER_SAVE_AUDIO = env.REPORT_BUFFER_SAVE_AUDIO;
  }

  if (env.STORAGE && env.R2_BUCKET_NAME && env.R2_ACCOUNT_ID && env.R2_ACCESS_KEY_ID && env.R2_SECRET_ACCESS_KEY) {
    vars.VOICESENTINEL_R2_BUCKET = env.R2_BUCKET_NAME;
    vars.R2_ENDPOINT_URL = `https://${env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`;
    vars.AWS_ACCESS_KEY_ID = env.R2_ACCESS_KEY_ID;
    vars.AWS_SECRET_ACCESS_KEY = env.R2_SECRET_ACCESS_KEY;
  }

  return vars;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const container = getContainer(env.PROCESSOR, "main");
    await container.startAndWaitForPorts({
      startOptions: { envVars: containerEnv(env) },
    });
    return container.fetch(request);
  },
};
