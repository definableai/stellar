import { lazy, Suspense, useEffect, type ComponentType } from "react";
import { BrowserRouter, Link, Route, Routes, useLocation } from "react-router-dom";
import Layout from "./layout/Layout";
import Page, { type PageModule } from "./layout/Page";
import { load } from "./pages";
import { idOf, name } from "./site";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="*" element={<Resolved />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

function Resolved() {
  const { pathname, hash } = useLocation();
  useScroll(pathname, hash);
  const View = view(pathname);
  if (!View) return <NotFound path={pathname} />;
  return (
    <Suspense fallback={<Skeleton />}>
      <View />
    </Suspense>
  );
}

const views = new Map<string, ComponentType>();

/** The lazy view of a page, made once: React.lazy must not run twice for one page. */
function view(path: string) {
  const loader = load(path);
  if (!loader) return undefined;
  const id = idOf(path);
  let made = views.get(id);
  if (!made) {
    made = lazy(async () => {
      const mod = (await loader()) as PageModule;
      return { default: () => <Page mod={mod} path={path} /> };
    });
    views.set(id, made);
  }
  return made;
}

/** Top of the page on a move, the heading when the URL names one. */
function useScroll(pathname: string, hash: string) {
  useEffect(() => {
    if (!hash) { window.scrollTo(0, 0); return; }
    let frame = 0;
    let tries = 0;
    const go = () => {
      const target = document.getElementById(decodeURIComponent(hash.slice(1)));
      if (target) target.scrollIntoView();
      // the page is lazy: wait for it to mount, but not forever (~1s).
      else if (tries++ < 60) frame = requestAnimationFrame(go);
    };
    go();
    return () => cancelAnimationFrame(frame);
  }, [pathname, hash]);
}

const Skeleton = () => (
  <div className="mx-auto max-w-[40.5rem] animate-pulse px-5 pt-8">
    <div className="h-9 w-1/2 rounded-md bg-gray-100 dark:bg-white/5" />
    <div className="mt-6 h-4 w-full rounded-md bg-gray-100 dark:bg-white/5" />
    <div className="mt-3 h-4 w-4/5 rounded-md bg-gray-100 dark:bg-white/5" />
  </div>
);

function NotFound({ path }: { path: string }) {
  useEffect(() => { document.title = `Page not found - ${name}`; }, []);
  return (
    <div className="mx-auto max-w-[40.5rem] px-5 pt-8 pb-16">
      <h1 className="text-[36px] leading-9 font-medium tracking-[-0.5px] text-gray-900 dark:text-gray-200">Page not found</h1>
      <p className="mt-3 text-[18px] leading-7 text-gray-500 dark:text-gray-400">
        Nothing is written at{" "}
        <code className="rounded-md bg-gray-950/5 px-2 py-0.5 font-mono text-sm text-gray-800 dark:bg-white/5 dark:text-gray-200">{path}</code>.
      </p>
      <Link
        to="/"
        className="mt-8 inline-block text-sm font-medium text-primary underline decoration-primary/40 underline-offset-4 hover:decoration-primary dark:text-primary-light dark:decoration-primary-light/40 dark:hover:decoration-primary-light"
      >
        Back to the introduction
      </Link>
    </div>
  );
}
