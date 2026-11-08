// Next remounts a template on every navigation, which replays the page's fade-in.
export default function Template({ children }) {
  return <div className="page">{children}</div>;
}
