#!/usr/bin/env node
/**
 * loom — multi-agent orchestration across git worktrees.
 *
 * Two modes:
 *   MCP server:  node index.js serve     (or no args — default for .mcp.json)
 *   CLI:         node index.js status
 *                node index.js review <lane> [--tests] [--stat] [--files]
 *                node index.js approve <lane> [-r]
 *
 * Aliases: `lr` = `loom review`, `la` = `loom approve`
 *
 * Configured per-project via `loom.json` at the git root.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

interface LoomConfig {
  approvalsDir: string;
  reviewsDir: string;
  ownershipFile: string;
  flightFile: string;
  branchPrefix: string;
  mainBranch: string;
}

const DEFAULTS: LoomConfig = {
  approvalsDir: "agent-orchestrator/approvals",
  reviewsDir: "agent-orchestrator/reviews",
  ownershipFile: "agent-orchestrator/ownership",
  flightFile: "agent-orchestrator/flight.md",
  branchPrefix: "agent/",
  mainBranch: "master",
};

function gitRoot(): string {
  return execSync("git rev-parse --show-toplevel", { encoding: "utf-8" }).trim();
}

function loadConfig(): LoomConfig {
  const configPath = path.join(gitRoot(), "loom.json");
  if (fs.existsSync(configPath)) {
    const raw = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    return { ...DEFAULTS, ...raw };
  }
  return DEFAULTS;
}

const config = loadConfig();
const root = gitRoot();

function resolve(p: string): string {
  return path.join(root, p);
}

function git(args: string): string {
  return execSync(`git ${args}`, { encoding: "utf-8", cwd: root }).trim();
}

function gitRaw(args: string): Buffer {
  return execSync(`git ${args}`, { cwd: root });
}

// ---------------------------------------------------------------------------
// Ownership
// ---------------------------------------------------------------------------

function allLanes(): string[] {
  const ownershipPath = resolve(config.ownershipFile);
  if (!fs.existsSync(ownershipPath)) return [];
  const lines = fs.readFileSync(ownershipPath, "utf-8").split("\n");
  const prefix = config.branchPrefix.replace(/\/$/, "");
  const re = new RegExp(`^${prefix}/(\\S+)\\s*=`);
  const lanes: string[] = [];
  for (const line of lines) {
    const match = line.match(re);
    if (match) lanes.push(match[1]);
  }
  return lanes;
}

function ownedPaths(lane: string): string[] {
  const ownershipPath = resolve(config.ownershipFile);
  if (!fs.existsSync(ownershipPath)) return [];
  const lines = fs.readFileSync(ownershipPath, "utf-8").split("\n");
  const prefix = config.branchPrefix.replace(/\/$/, "");
  const key = `${prefix}/${lane}`;
  for (const line of lines) {
    if (line.startsWith("#") || !line.trim()) continue;
    const [k, v] = line.split("=", 2);
    if (k.trim() === key && v) {
      return v.trim().split(/\s+/);
    }
  }
  return [];
}

// ---------------------------------------------------------------------------
// Approvals
// ---------------------------------------------------------------------------

interface ApprovalState {
  lane: string;
  approved: boolean;
  sha?: string;
  branchHead?: string;
  stale: boolean;
}

function branchHead(lane: string): string | undefined {
  try {
    return git(`rev-parse ${config.branchPrefix}${lane}`);
  } catch {
    return undefined;
  }
}

function checkLane(lane: string): ApprovalState {
  const file = path.join(resolve(config.approvalsDir), lane);
  const head = branchHead(lane);

  if (!fs.existsSync(file)) {
    return { lane, approved: false, branchHead: head, stale: false };
  }

  const sha = fs.readFileSync(file, "utf-8").trim();
  const stale = head !== undefined && sha !== head;

  return { lane, approved: !stale, sha, branchHead: head, stale };
}

// ---------------------------------------------------------------------------
// Reviews
// ---------------------------------------------------------------------------

type ReviewResult = "pass" | "fail" | "none";

function checkReview(lane: string): ReviewResult {
  const file = path.join(resolve(config.reviewsDir), lane);
  if (!fs.existsSync(file)) return "none";
  const content = fs.readFileSync(file, "utf-8").trim().toUpperCase();
  if (content === "PASS") return "pass";
  if (content === "FAIL") return "fail";
  return "none";
}

// ---------------------------------------------------------------------------
// Status board
// ---------------------------------------------------------------------------

interface LaneInfo {
  lane: string;
  status: string;
  model: string;
  doing: string;
  commitsAhead: number;
  hasWorktree: boolean;
  review: ReviewResult;
  approval: ApprovalState;
}

function parseFlight(): Map<string, { status: string; model: string; doing: string }> {
  const flightPath = resolve(config.flightFile);
  if (!fs.existsSync(flightPath)) return new Map();
  const content = fs.readFileSync(flightPath, "utf-8");
  const lines = content.split("\n");
  const result = new Map<string, { status: string; model: string; doing: string }>();

  let inLanes = false;
  for (const line of lines) {
    if (line.includes("lanes:begin")) { inLanes = true; continue; }
    if (line.includes("lanes:end")) { inLanes = false; continue; }
    if (!inLanes) continue;
    if (line.startsWith("<!--") || line.startsWith("| ---")) continue;

    const match = line.match(/^\|\s*`([^`]+)`\s*\|\s*([^|]+)\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|/);
    if (match) {
      result.set(match[1].trim(), {
        status: match[2].trim(),
        model: match[4].trim(),
        doing: match[5].trim(),
      });
    }
  }
  return result;
}

function commitsAhead(lane: string): number {
  try {
    return parseInt(
      git(`rev-list --count ${config.mainBranch}..${config.branchPrefix}${lane}`),
      10,
    );
  } catch {
    return 0;
  }
}

function laneHasWorktree(lane: string): boolean {
  try {
    const output = git("worktree list --porcelain");
    return output.includes(`branch refs/heads/${config.branchPrefix}${lane}`);
  } catch {
    return false;
  }
}

function getLaneInfo(
  lane: string,
  flight: Map<string, { status: string; model: string; doing: string }>,
): LaneInfo {
  const flightData = flight.get(lane) ?? { status: "unknown", model: "—", doing: "—" };
  return {
    lane,
    status: flightData.status,
    model: flightData.model,
    doing: flightData.doing,
    commitsAhead: commitsAhead(lane),
    hasWorktree: laneHasWorktree(lane),
    review: checkReview(lane),
    approval: checkLane(lane),
  };
}

function pad(s: string, n: number): string {
  return s.length >= n ? s : s + " ".repeat(n - s.length);
}

function formatStatus(infos: LaneInfo[]): string {
  const lines: string[] = [];
  lines.push(
    `${pad("Lane", 16)} ${pad("Status", 20)} ${pad("Ahead", 6)} ${pad("Worktree", 10)} ${pad("Reviewed", 10)} ${pad("Approved", 10)} Model`,
  );
  lines.push("-".repeat(90));

  for (const info of infos) {
    if (info.status === "merged") continue;
    const reviewStr = info.review === "pass" ? "pass" : info.review === "fail" ? "FAIL" : "—";
    const approvalStr = info.approval.approved
      ? "yes"
      : info.approval.stale
        ? "stale"
        : "no";
    const worktreeStr = info.hasWorktree ? "yes" : "—";

    lines.push(
      `${pad(info.lane, 16)} ${pad(info.status, 20)} ${pad(String(info.commitsAhead), 6)} ${pad(worktreeStr, 10)} ${pad(reviewStr, 10)} ${pad(approvalStr, 10)} ${info.model}`,
    );
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Flight log updates
// ---------------------------------------------------------------------------

const VALID_STATUSES = [
  "queued",
  "in-flight",
  "blocked",
  "awaiting-review",
  "in-review",
  "changes-requested",
  "merged",
  "abandoned",
] as const;

type LaneStatus = (typeof VALID_STATUSES)[number];

interface FlightUpdate {
  lane: string;
  status?: LaneStatus;
  agent?: string;
  model?: string;
  doing?: string;
  blocker?: string;
  log?: string;
}

function updateFlight(update: FlightUpdate): string {
  const flightPath = resolve(config.flightFile);
  if (!fs.existsSync(flightPath)) {
    throw new Error(`flight file not found: ${config.flightFile}`);
  }

  const content = fs.readFileSync(flightPath, "utf-8");
  const lines = content.split("\n");

  let found = false;
  let inLanes = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.includes("lanes:begin")) { inLanes = true; continue; }
    if (line.includes("lanes:end")) { inLanes = false; continue; }
    if (!inLanes) continue;
    if (line.startsWith("<!--") || line.startsWith("| ---") || line.startsWith("| Lane")) continue;

    const match = line.match(/^\|\s*`([^`]+)`\s*\|/);
    if (match && match[1].trim() === update.lane) {
      // Parse existing columns: Lane | Status | Agent | Model | Doing | Blocker |
      const cols = line.split("|").slice(1, -1); // drop leading/trailing empty
      if (cols.length < 6) continue;

      const cur = {
        lane: cols[0].trim(),
        status: cols[1].trim(),
        agent: cols[2].trim(),
        model: cols[3].trim(),
        doing: cols[4].trim(),
        blocker: cols[5].trim(),
      };

      const newStatus = update.status ?? cur.status;
      const newAgent = update.agent ?? cur.agent;
      const newModel = update.model ?? cur.model;
      const newDoing = update.doing ?? cur.doing;
      const newBlocker = update.blocker ?? cur.blocker;

      lines[i] = `| \`${update.lane}\` | ${newStatus} | ${newAgent} | ${newModel} | ${newDoing} | ${newBlocker} |`;
      found = true;
      break;
    }
  }

  if (!found) {
    throw new Error(`lane '${update.lane}' not found in flight log`);
  }

  // Append log entry if provided
  if (update.log) {
    const today = new Date().toISOString().slice(0, 10);
    const logEntry = `- \`${today}\` — ${update.log}`;

    // Find last non-empty line and append
    let lastNonEmpty = lines.length - 1;
    while (lastNonEmpty >= 0 && !lines[lastNonEmpty].trim()) lastNonEmpty--;
    lines.splice(lastNonEmpty + 1, 0, logEntry);
  }

  fs.writeFileSync(flightPath, lines.join("\n"));

  const summary = [
    `updated ${update.lane}`,
    update.status ? `status → ${update.status}` : null,
    update.agent ? `agent → ${update.agent}` : null,
    update.model ? `model → ${update.model}` : null,
    update.blocker ? `blocker → ${update.blocker}` : null,
    update.log ? `log entry added` : null,
  ]
    .filter(Boolean)
    .join(", ");

  return summary;
}

// ---------------------------------------------------------------------------
// CLI: review
// ---------------------------------------------------------------------------

function cliReview(lane: string, args: string[]): void {
  const branch = `${config.branchPrefix}${lane}`;
  try {
    git(`rev-parse --verify ${branch}`);
  } catch {
    console.error(`error: branch '${branch}' does not exist`);
    process.exit(1);
  }

  let showTests = false;
  const gitArgs: string[] = [];
  for (const arg of args) {
    if (arg === "--tests") showTests = true;
    else if (arg === "--stat") gitArgs.push("--stat");
    else if (arg === "--files") gitArgs.push("--name-only");
    else gitArgs.push(arg);
  }

  let paths = ownedPaths(lane);
  if (!showTests) {
    paths = paths.filter((p) => !p.endsWith("_test.rs"));
  }

  const extra = gitArgs.length ? " " + gitArgs.join(" ") : "";
  const pathSpec = paths.length ? " -- " + paths.join(" ") : "";

  try {
    const output = gitRaw(`diff ${config.mainBranch}..${branch}${extra}${pathSpec}`);
    process.stdout.write(output);
  } catch {
    // git diff returns exit 1 when there are differences — that's fine
  }
}

// ---------------------------------------------------------------------------
// CLI: approve
// ---------------------------------------------------------------------------

function cliApprove(lane: string, args: string[]): void {
  const branch = `${config.branchPrefix}${lane}`;
  try {
    git(`rev-parse --verify ${branch}`);
  } catch {
    console.error(`error: branch '${branch}' does not exist`);
    process.exit(1);
  }

  const dir = resolve(config.approvalsDir);
  fs.mkdirSync(dir, { recursive: true });

  // Revoke mode
  if (args.includes("-r")) {
    const file = path.join(dir, lane);
    if (fs.existsSync(file)) fs.unlinkSync(file);
    console.log(`revoked approval for ${lane}`);
    return;
  }

  const sha = git(`rev-parse ${branch}`);
  const stat = git(`diff --stat ${config.mainBranch}..${branch}`);

  fs.writeFileSync(path.join(dir, lane), sha + "\n");

  console.log(`approved ${lane} at ${sha}`);
  console.log(stat);
}

// ---------------------------------------------------------------------------
// CLI: status
// ---------------------------------------------------------------------------

function cliStatus(): void {
  const flight = parseFlight();
  const lanes = allLanes();
  const infos = lanes.map((l) => getLaneInfo(l, flight));
  console.log(formatStatus(infos));
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

async function serveMcp(): Promise<void> {
  const { McpServer } = await import("@modelcontextprotocol/sdk/server/mcp.js");
  const { StdioServerTransport } = await import("@modelcontextprotocol/sdk/server/stdio.js");
  const { z } = await import("zod");

  const server = new McpServer({
    name: "loom",
    version: "0.1.0",
  });

  server.tool(
    "status",
    "Full orchestrator board: every lane with its flight-log status, commits ahead of main, worktree presence, approval state, and model. Always show the result to the user.",
    {},
    async () => {
      const flight = parseFlight();
      const lanes = allLanes();
      const infos = lanes.map((l) => getLaneInfo(l, flight));
      return { content: [{ type: "text", text: formatStatus(infos) }] };
    },
  );

  server.tool(
    "check_approvals",
    "Check human-approval state for all lanes (or a specific lane). Returns which lanes are approved at their branch HEAD, which are stale, and which are pending.",
    {
      lane: z.string().optional().describe("Check a specific lane (omit for all)"),
    },
    async ({ lane }) => {
      const lanes = lane ? [lane] : allLanes();
      const results = lanes.map(checkLane);

      const text = results
        .map((r) => {
          if (r.approved) return `✓ ${r.lane} approved at ${r.sha?.slice(0, 7)}`;
          if (r.stale)
            return `⚠ ${r.lane} approval stale (approved ${r.sha?.slice(0, 7)}, HEAD ${r.branchHead?.slice(0, 7)})`;
          return `✗ ${r.lane} not approved (HEAD ${r.branchHead?.slice(0, 7) ?? "no branch"})`;
        })
        .join("\n");

      return { content: [{ type: "text", text }] };
    },
  );

  server.tool(
    "wait_for_approval",
    "Block until a new approval file is created or modified. Returns the lane name and SHA. Call this from a background agent to get zero-latency notification when the user approves a lane. Times out after the specified seconds (default 600).",
    {
      timeout_seconds: z
        .number()
        .default(600)
        .describe("Max seconds to wait (default 600 = 10 min)"),
    },
    async ({ timeout_seconds }) => {
      const dir = resolve(config.approvalsDir);
      fs.mkdirSync(dir, { recursive: true });

      const baseline = new Map<string, number>();
      for (const f of fs.readdirSync(dir)) {
        try {
          baseline.set(f, fs.statSync(path.join(dir, f)).mtimeMs);
        } catch { /* deleted between readdir and stat */ }
      }

      return new Promise((resolve) => {
        const timer = setTimeout(() => {
          watcher.close();
          resolve({
            content: [{ type: "text", text: "timeout: no approval received" }],
          });
        }, timeout_seconds * 1000);

        const watcher = fs.watch(dir, (event, filename) => {
          if (!filename) return;
          if (event !== "rename" && event !== "change") return;

          const filePath = path.join(dir, filename);
          try {
            const mtime = fs.statSync(filePath).mtimeMs;
            const prev = baseline.get(filename);
            if (prev !== undefined && mtime <= prev) return;
          } catch { return; }

          const lane = filename;
          const state = checkLane(lane);

          clearTimeout(timer);
          watcher.close();

          resolve({
            content: [
              {
                type: "text",
                text: state.approved
                  ? `${lane} approved at ${state.sha?.slice(0, 7)}`
                  : `${lane} approval file changed but stale (SHA mismatch)`,
              },
            ],
          });
        });
      });
    },
  );

  server.tool(
    "update_flight",
    "Update a lane's status and metadata in the flight log. Use this to record state transitions (e.g. queued → in-flight, in-flight → awaiting-review, awaiting-review → merged). Optionally appends an audit log entry.",
    {
      lane: z.string().describe("Lane name (e.g. 'attention')"),
      status: z
        .enum(VALID_STATUSES)
        .optional()
        .describe("New status for the lane"),
      agent: z.string().optional().describe("Agent identifier"),
      model: z.string().optional().describe("Model used (e.g. 'opus', 'sonnet')"),
      doing: z.string().optional().describe("Short description of current work"),
      blocker: z.string().optional().describe("Blocker description, or '—' to clear"),
      log: z.string().optional().describe("Audit log entry to append (date is added automatically)"),
    },
    async ({ lane, status, agent, model, doing, blocker, log }) => {
      try {
        const summary = updateFlight({ lane, status, agent, model, doing, blocker, log });
        return { content: [{ type: "text", text: summary }] };
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        return { content: [{ type: "text", text: `error: ${msg}` }], isError: true };
      }
    },
  );

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

const [cmd, ...rest] = process.argv.slice(2);

switch (cmd) {
  case "status":
  case "s":
    cliStatus();
    break;

  case "review":
  case "r": {
    const lane = rest[0];
    if (!lane) { console.error("usage: loom review <lane> [--tests] [--stat] [--files]"); process.exit(1); }
    cliReview(lane, rest.slice(1));
    break;
  }

  case "approve":
  case "a": {
    const lane = rest[0];
    if (!lane) { console.error("usage: loom approve <lane> [-r]"); process.exit(1); }
    cliApprove(lane, rest.slice(1));
    break;
  }

  case "flight":
  case "f": {
    // loom flight <lane> <status> [--model X] [--agent X] [--blocker X] [--log "msg"]
    const lane = rest[0];
    const status = rest[1] as LaneStatus | undefined;
    if (!lane) {
      console.error("usage: loom flight <lane> [status] [--model M] [--agent A] [--blocker B] [--log MSG]");
      process.exit(1);
    }
    if (status && !VALID_STATUSES.includes(status)) {
      console.error(`invalid status: ${status}`);
      console.error(`valid: ${VALID_STATUSES.join(", ")}`);
      process.exit(1);
    }

    const flags: Record<string, string> = {};
    for (let i = 2; i < rest.length; i++) {
      if (rest[i].startsWith("--") && i + 1 < rest.length) {
        flags[rest[i].slice(2)] = rest[i + 1];
        i++;
      }
    }

    try {
      const summary = updateFlight({
        lane,
        status,
        agent: flags.agent,
        model: flags.model,
        blocker: flags.blocker,
        doing: flags.doing,
        log: flags.log,
      });
      console.log(summary);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error(`error: ${msg}`);
      process.exit(1);
    }
    break;
  }

  case "serve":
  case undefined:
    // Default: MCP server mode (for .mcp.json)
    serveMcp();
    break;

  default:
    console.error(`unknown command: ${cmd}`);
    console.error("usage: loom [status|review|approve|serve]");
    process.exit(1);
}
