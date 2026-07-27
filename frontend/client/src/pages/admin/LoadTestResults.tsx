export default function LoadTestResults() {
  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-slow">
      <h1 className="text-2xl font-bold">Load Test Results</h1>
      <p className="text-secondary text-body max-w-2xl">
        View Locust stress test outputs for the FastAPI backend (WIP).
      </p>
    </div>
  );
}
