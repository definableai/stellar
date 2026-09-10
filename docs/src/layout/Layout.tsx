import type { ReactNode } from "react";
import Sidebar from "./Sidebar";
import Toc from "./Toc";
import Topbar from "./Topbar";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <>
      <Topbar />
      <div className="mx-auto grid max-w-[90rem] lg:grid-cols-[224px_minmax(0,1fr)] xl:grid-cols-[224px_minmax(0,1fr)_236px]">
        <Sidebar />
        <main id="content">{children}</main>
        <Toc />
      </div>
    </>
  );
}
